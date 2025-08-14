from ultralytics import YOLO
import cv2, os, shutil
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from collections import deque

class AdaptiveYOLOInference:
    def __init__(self, model_path, min_workers=1, max_workers=8, queue_size=10):
        self.model_path = model_path
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 創建隊列 - 增加緩衝區大小
        self.frame_queue = asyncio.Queue(maxsize=queue_size * 2)
        self.result_queue = asyncio.Queue(maxsize=queue_size)
        
        # 模型實例字典（線程安全）
        self.models = {}
        self.models_lock = threading.Lock()
        
        # *** 預載入模型池 ***
        self._preload_models()
        
        # 工作者管理
        self.workers = {}
        self.worker_counter = 0
        self.workers_lock = asyncio.Lock()
        
        # 性能監控
        self.performance_stats = {
            'queue_lengths': deque(maxlen=50),
            'processing_times': deque(maxlen=50),
            'worker_utilization': {},
            'last_adjustment': time.time(),
            'adjustment_cooldown': 3.0  # 增加調整冷卻時間
        }
        
        self.should_stop = False
        
    def _preload_models(self):
        """預載入模型池以避免運行時載入延遲"""
        print("Preloading YOLO models...")
        
        def load_model(i):
            # 為每個預期的執行緒預載入模型
            dummy_thread_id = f"preload_{i}"
            model = YOLO(self.model_path, task="detect")
            # 預熱模型
            import numpy as np
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            model.predict(dummy_input, verbose=False)
            return dummy_thread_id, model
        
        # 預載入最大工作者數量的模型
        with ThreadPoolExecutor(max_workers=4) as preload_executor:
            futures = [preload_executor.submit(load_model, i) for i in range(self.max_workers)]
            for future in futures:
                thread_id, model = future.result()
                self.models[thread_id] = model
        
        print(f"Preloaded {len(self.models)} YOLO models")
        
    def get_model_for_thread(self):
        """為當前線程獲取模型實例（使用預載入模型）"""
        thread_id = threading.current_thread().ident
        
        with self.models_lock:
            if thread_id not in self.models:
                # 嘗試重用預載入的模型
                preload_keys = [k for k in self.models.keys() if str(k).startswith("preload_")]
                if preload_keys:
                    # 重用預載入模型
                    preload_key = preload_keys[0]
                    self.models[thread_id] = self.models.pop(preload_key)
                else:
                    # 如果沒有預載入模型，才創建新的
                    self.models[thread_id] = YOLO(self.model_path, task="detect")
        
        return self.models[thread_id]
    
    def predict_frame(self, frame_data):
        """在線程池中執行的推理函數"""
        frame, frame_id, worker_id = frame_data
        
        model = self.get_model_for_thread()
        
        start_time = time.time()
        result = model.predict(frame, verbose=False)[0]
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.performance_stats['processing_times'].append(processing_time)
        
        return result, frame_id, processing_time

    async def frame_producer(self, cap):
        """生產者：讀取視頻幀並放入隊列"""
        frame_id = 0
        frame_skip = 0  # 動態跳幀
        
        while not self.should_stop:
            ret, frame = cap.read()
            if not ret:
                self.should_stop = True
                break
            
            frame_id += 1
            
            # 動態跳幀邏輯
            queue_size = self.frame_queue.qsize()
            if queue_size > self.frame_queue.maxsize * 0.8:
                frame_skip = min(frame_skip + 1, 3)
            else:
                frame_skip = max(frame_skip - 1, 0)
            
            if frame_skip > 0 and frame_id % frame_skip == 0:
                continue  # 跳過此幀
            
            try:
                self.performance_stats['queue_lengths'].append(queue_size)
                await self.frame_queue.put((frame, frame_id))
                
            except asyncio.QueueFull:
                continue
                
            # 減少讓出頻率
            if frame_id % 5 == 0:
                await asyncio.sleep(0.001)
        
        # 發送結束信號
        for _ in range(self.max_workers):
            await self.frame_queue.put(None)
    
    async def inference_worker(self, worker_id):
        """推理工作者"""
        idle_start = None
        consecutive_errors = 0
        
        try:
            while not self.should_stop:
                try:
                    frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=2.0)
                    
                    if frame_data is None:
                        break
                    
                    frame, frame_id = frame_data
                    idle_start = None
                    consecutive_errors = 0
                    
                    loop = asyncio.get_event_loop()
                    result, result_frame_id, processing_time = await loop.run_in_executor(
                        self.executor, self.predict_frame, (frame, frame_id, worker_id)
                    )
                    
                    await self.result_queue.put((result, result_frame_id, frame, processing_time))
                    self.frame_queue.task_done()
                    
                except asyncio.TimeoutError:
                    if idle_start is None:
                        idle_start = time.time()
                    elif time.time() - idle_start > 10.0:  # 增加空閒超時時間
                        break
                        
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors > 5:  # 連續錯誤太多就退出
                        break
                    continue
                    
        finally:
            async with self.workers_lock:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                    self.current_workers -= 1
    
    async def adaptive_worker_manager(self):
        """改進的動態工作者管理器"""
        while not self.should_stop:
            await asyncio.sleep(2.0)
            
            queue_lengths = list(self.performance_stats['queue_lengths'])
            processing_times = list(self.performance_stats['processing_times'])
            
            if len(queue_lengths) < 10:
                continue
            
            avg_queue_length = sum(queue_lengths[-10:]) / len(queue_lengths[-10:])
            avg_processing_time = sum(processing_times[-10:]) / len(processing_times[-10:]) if processing_times else 0
            
            current_time = time.time()
            time_since_last_adjustment = current_time - self.performance_stats['last_adjustment']
            
            # 使用冷卻時間避免頻繁調整
            if time_since_last_adjustment < self.performance_stats['adjustment_cooldown']:
                continue
            
            async with self.workers_lock:
                # 更智能的調整邏輯
                if avg_queue_length > 8.0 and self.current_workers < self.max_workers:
                    # 只有在隊列真的很滿時才增加工作者
                    await self._add_worker()
                    self.performance_stats['last_adjustment'] = current_time
                elif avg_queue_length < 2.0 and self.current_workers > self.min_workers:
                    # 減少工作者的邏輯
                    self._request_worker_shutdown()
                    self.performance_stats['last_adjustment'] = current_time
    
    def _request_worker_shutdown(self):
        """請求關閉一個工作者"""
        # 這裡只是標記，實際的關閉由工作者的空閒超時處理
        pass
    
    async def _add_worker(self):
        """添加新的工作者"""
        self.worker_counter += 1
        worker_id = self.worker_counter
        
        worker_task = asyncio.create_task(self.inference_worker(worker_id))
        self.workers[worker_id] = worker_task
        self.current_workers += 1
    
    async def result_consumer(self):
        """消費者：顯示推理結果"""
        processed_frames = {}
        next_display_frame = 1
        display_skip = 0  # 顯示跳幀
        
        while not self.should_stop:
            try:
                result_data = await asyncio.wait_for(self.result_queue.get(), timeout=3.0)
                
                if result_data is None:
                    break
                
                result, frame_id, original_frame, processing_time = result_data
                processed_frames[frame_id] = (result, original_frame)
                
                # 按順序顯示結果，但允許跳幀
                while next_display_frame in processed_frames:
                    display_result, display_frame = processed_frames.pop(next_display_frame)
                    
                    # 動態顯示跳幀
                    if len(processed_frames) > 10:
                        display_skip = 2
                    elif len(processed_frames) > 5:
                        display_skip = 1
                    else:
                        display_skip = 0
                    
                    if display_skip == 0 or next_display_frame % (display_skip + 1) == 0:
                        display_img = cv2.resize(display_result.plot(), (720, 480))
                        cv2.imshow("Adaptive YOLO Inference", display_img)
                    
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.should_stop = True
                        return
                    
                    next_display_frame += 1
                
                self.result_queue.task_done()
                
            except asyncio.TimeoutError:
                if self.frame_queue.empty() and self.result_queue.empty() and self.should_stop:
                    break

    # run_inference 方法保持不變
    async def run_inference(self, video_path):
        """主要的推理流程"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        try:
            async with self.workers_lock:
                for i in range(self.min_workers):
                    await self._add_worker()
            
            tasks = [
                asyncio.create_task(self.frame_producer(cap)),
                asyncio.create_task(self.result_consumer()),
                asyncio.create_task(self.adaptive_worker_manager())
            ]
            
            async with self.workers_lock:
                for worker_task in self.workers.values():
                    tasks.append(worker_task)
            
            await asyncio.gather(*tasks[:3])
            
            async with self.workers_lock:
                remaining_workers = list(self.workers.values())
            
            if remaining_workers:
                await asyncio.gather(*remaining_workers, return_exceptions=True)
            
        finally:
            self.should_stop = True
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=True)

# main 函數保持不變
async def main():
    parser = argparse.ArgumentParser(description="自適應 YOLO 並發推理")
    parser.add_argument("--video_path", type=str, default="./data/video.mp4")
    parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
    parser.add_argument("--min_workers", type=int, default=2, help="最小工作者數量")
    parser.add_argument("--max_workers", type=int, default=6, help="最大工作者數量")
    parser.add_argument("--queue_size", type=int, default=32, help="幀隊列大小")
    args = parser.parse_args()
    
    bin_dir = './bin'
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir)
    os.mkdir(bin_dir)
    
    inference_system = AdaptiveYOLOInference(
        model_path=args.tflite_model,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        queue_size=args.queue_size
    )
    
    await inference_system.run_inference(args.video_path)

if __name__ == "__main__":
    asyncio.run(main())
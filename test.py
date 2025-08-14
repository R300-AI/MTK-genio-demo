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
        
        # 創建隊列
        self.frame_queue = asyncio.Queue(maxsize=queue_size)
        self.result_queue = asyncio.Queue()
        
        # 模型實例字典（線程安全）
        self.models = {}
        self.models_lock = threading.Lock()
        
        # 工作者管理
        self.workers = {}  # worker_id -> task
        self.worker_counter = 0
        self.workers_lock = asyncio.Lock()
        
        # 性能監控
        self.performance_stats = {
            'queue_lengths': deque(maxlen=50),  # 隊列長度歷史
            'processing_times': deque(maxlen=50),  # 處理時間歷史
            'worker_utilization': {},  # 各工作者利用率
            'last_adjustment': time.time()
        }
        
        # 控制標誌
        self.should_stop = False
        
    def get_model_for_thread(self):
        """為當前線程獲取模型實例（線程安全）"""
        thread_id = threading.current_thread().ident
        with self.models_lock:
            if thread_id not in self.models:
                #print(f"[模型] 為線程 {thread_id} 創建新的模型實例")
                self.models[thread_id] = YOLO(self.model_path, task="detect")
        
        return self.models[thread_id]
    
    def predict_frame(self, frame_data):
        """在線程池中執行的推理函數"""
        frame, frame_id, worker_id = frame_data
        
        model = self.get_model_for_thread()
        thread_id = threading.current_thread().ident
        
        start_time = time.time()
        #print(f"[推理] 工作者 {worker_id} (線程 {thread_id}) 開始處理幀 {frame_id}")
        
        result = model.predict(frame, verbose=False)[0]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 記錄性能數據
        self.performance_stats['processing_times'].append(processing_time)
        
        #print(f"[推理] 工作者 {worker_id} 完成幀 {frame_id}，耗時 {processing_time:.3f}s")
        
        return result, frame_id, processing_time

    async def frame_producer(self, cap):
        """生產者：讀取視頻幀並放入隊列"""
        #print("[生產者] 開始讀取視頻幀")
        frame_id = 0
        
        while not self.should_stop:
            ret, frame = cap.read()
            if not ret:
                #print("[生產者] 視頻結束")
                self.should_stop = True
                break
            
            frame_id += 1
            #print(f"[生產者] 讀取幀 {frame_id}")
            
            try:
                # 記錄隊列長度用於動態調整
                queue_size = self.frame_queue.qsize()
                self.performance_stats['queue_lengths'].append(queue_size)
                
                await self.frame_queue.put((frame, frame_id))
                
            except asyncio.QueueFull:
                print(f"[生產者] 隊列已滿，跳過幀 {frame_id}")
                
            # 短暫讓出控制權
            await asyncio.sleep(0.001)
        
        # 發送結束信號
        #print("[生產者] 發送結束信號")
        for _ in range(self.max_workers):
            await self.frame_queue.put(None)
    
    async def inference_worker(self, worker_id):
        """推理工作者"""
        #print(f"[工作者 {worker_id}] 啟動")
        idle_start = None
        
        try:
            while not self.should_stop:
                try:
                    # 等待幀數據，設置超時避免永遠阻塞
                    frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=2.0)
                    
                    if frame_data is None:
                        #print(f"[工作者 {worker_id}] 收到結束信號")
                        break
                    
                    frame, frame_id = frame_data
                    
                    # 重置空閒時間
                    idle_start = None
                    
                    #print(f"[工作者 {worker_id}] 取得幀 {frame_id}")
                    
                    # 異步執行推理
                    loop = asyncio.get_event_loop()
                    result, result_frame_id, processing_time = await loop.run_in_executor(
                        self.executor, self.predict_frame, (frame, frame_id, worker_id)
                    )
                    
                    # 將結果放入結果隊列
                    await self.result_queue.put((result, result_frame_id, frame, processing_time))
                    
                    # 標記任務完成
                    self.frame_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # 工作者空閒
                    if idle_start is None:
                        idle_start = time.time()
                        #print(f"[工作者 {worker_id}] 開始空閒")
                    elif time.time() - idle_start > 5.0:  # 空閒超過5秒
                        #print(f"[工作者 {worker_id}] 空閒過久，準備關閉")
                        break
                        
                except Exception as e:
                    print(f"[工作者 {worker_id}] 發生錯誤: {e}")
                    
        finally:
            # 清理工作者
            async with self.workers_lock:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                    self.current_workers -= 1
                    #print(f"[管理] 工作者 {worker_id} 已關閉，當前工作者數: {self.current_workers}")
    
    async def adaptive_worker_manager(self):
        """動態工作者管理器"""
        #print("[管理器] 動態工作者管理器啟動")
        
        while not self.should_stop:
            await asyncio.sleep(2.0)  # 每2秒檢查一次
            
            # 獲取性能指標
            queue_lengths = list(self.performance_stats['queue_lengths'])
            processing_times = list(self.performance_stats['processing_times'])
            
            if len(queue_lengths) < 5:  # 數據不足，跳過調整
                continue
            
            avg_queue_length = sum(queue_lengths[-10:]) / len(queue_lengths[-10:])
            avg_processing_time = sum(processing_times[-10:]) / len(processing_times[-10:]) if processing_times else 0
            
            current_time = time.time()
            time_since_last_adjustment = current_time - self.performance_stats['last_adjustment']
            
            #print(f"[管理器] 平均隊列長度: {avg_queue_length:.1f}, 平均處理時間: {avg_processing_time:.3f}s")
            #print(f"[管理器] 當前工作者數: {self.current_workers}")
            
            should_adjust = False
            
            # 決策邏輯
            if time_since_last_adjustment >= 5.0:  # 至少5秒間隔調整
                async with self.workers_lock:
                    if avg_queue_length > 5.0 and self.current_workers < self.max_workers:
                        # 隊列積壓，需要增加工作者
                        #print(f"[管理器] 隊列積壓 ({avg_queue_length:.1f})，增加工作者")
                        await self._add_worker()
                        should_adjust = True
                        
                    elif avg_queue_length < 1.0 and self.current_workers > self.min_workers:
                        # 隊列空閒，可以減少工作者（但不主動減少，讓空閒超時機制處理）
                        print(f"[管理器] 隊列空閒 ({avg_queue_length:.1f})，工作者將自動減少")
                        
                    if should_adjust:
                        self.performance_stats['last_adjustment'] = current_time
    
    async def _add_worker(self):
        """添加新的工作者"""
        self.worker_counter += 1
        worker_id = self.worker_counter
        
        worker_task = asyncio.create_task(self.inference_worker(worker_id))
        self.workers[worker_id] = worker_task
        self.current_workers += 1
        
        #print(f"[管理] 新增工作者 {worker_id}，當前工作者數: {self.current_workers}")
    
    async def result_consumer(self):
        """消費者：顯示推理結果"""
        #print("[消費者] 開始顯示結果")
        processed_frames = {}
        next_display_frame = 1
        
        while not self.should_stop:
            try:
                result_data = await asyncio.wait_for(self.result_queue.get(), timeout=3.0)
                
                if result_data is None:
                    #print("[消費者] 收到結束信號")
                    break
                
                result, frame_id, original_frame, processing_time = result_data
                #print(f"[消費者] 收到幀 {frame_id} 的推理結果 (處理時間: {processing_time:.3f}s)")
                
                # 暫存結果
                processed_frames[frame_id] = (result, original_frame)
                
                # 按順序顯示結果
                while next_display_frame in processed_frames:
                    display_result, display_frame = processed_frames.pop(next_display_frame)
                    
                    #print(f"[消費者] 顯示幀 {next_display_frame}")
                    display_img = cv2.resize(display_result.plot(), (720, 480))
                    cv2.imshow("Adaptive YOLO Inference", display_img)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        #print("[消費者] 偵測到 ESC 鍵，準備退出")
                        self.should_stop = True
                        return
                    
                    next_display_frame += 1
                
                self.result_queue.task_done()
                
            except asyncio.TimeoutError:
                if self.frame_queue.empty() and self.result_queue.empty() and self.should_stop:
                    break
    
    async def run_inference(self, video_path):
        """主要的推理流程"""
        #print(f"[主流程] 開始自適應處理視頻: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            #print(f"[錯誤] 無法打開視頻文件: {video_path}")
            return
        
        try:
            # 初始化最小數量的工作者
            async with self.workers_lock:
                for i in range(self.min_workers):
                    await self._add_worker()
            
            # 創建任務
            tasks = [
                asyncio.create_task(self.frame_producer(cap)),
                asyncio.create_task(self.result_consumer()),
                asyncio.create_task(self.adaptive_worker_manager())
            ]
            
            # 添加初始工作者任務
            async with self.workers_lock:
                for worker_task in self.workers.values():
                    tasks.append(worker_task)
            
            #print(f"[主流程] 啟動 {len(tasks)} 個初始任務")
            
            # 等待主要任務完成
            await asyncio.gather(*tasks[:3])  # 只等待固定的3個主任務
            
            # 等待所有工作者完成
            async with self.workers_lock:
                remaining_workers = list(self.workers.values())
            
            if remaining_workers:
                #print(f"[主流程] 等待 {len(remaining_workers)} 個工作者完成")
                await asyncio.gather(*remaining_workers, return_exceptions=True)
            
        finally:
            #print("[主流程] 清理資源")
            self.should_stop = True
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=True)

async def main():
    print("[階段] 解析命令列參數")
    parser = argparse.ArgumentParser(description="自適應 YOLO 並發推理")
    parser.add_argument("--video_path", type=str, default="./data/video.mp4")
    parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
    parser.add_argument("--min_workers", type=int, default=1, help="最小工作者數量")
    parser.add_argument("--max_workers", type=int, default=10, help="最大工作者數量")
    parser.add_argument("--queue_size", type=int, default=64, help="幀隊列大小")
    args = parser.parse_args()
    
    print(f"[設定] 視頻路徑: {args.video_path}")
    print(f"[設定] 模型路徑: {args.tflite_model}")
    print(f"[設定] 工作者範圍: {args.min_workers} - {args.max_workers}")
    print(f"[設定] 隊列大小: {args.queue_size}")
    bin_dir = './bin'
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir)
    os.mkdir(bin_dir)
    
    # 創建自適應推理實例
    inference_system = AdaptiveYOLOInference(
        model_path=args.tflite_model,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        queue_size=args.queue_size
    )
    
    # 執行推理
    await inference_system.run_inference(args.video_path)

if __name__ == "__main__":
    print("[程序] 啟動自適應 YOLO 推理系統")
    asyncio.run(main())

from ultralytics import YOLO
import cv2, os, shutil
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from collections import deque
import numpy as np
import logging
import warnings

# 取消警告
warnings.filterwarnings("ignore")

# 配置全局日誌 - 統一輸出到 performance_stats.txt，自動清空
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台輸出
        logging.FileHandler('performance_stats.txt', mode='w', encoding='utf-8')  # 自動清空並寫入
    ]
)

# 創建主日誌記錄器
logger = logging.getLogger('ultralytics_demo')

class InferencePipeline:
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
        
        # 記錄初始化信息
        self._log_initialization()
        
        self.should_stop = False
        
    def _log_initialization(self):
        """記錄初始化信息"""
        logger.info(f"=== Performance Stats Started ===")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Min Workers: {self.min_workers}, Max Workers: {self.max_workers}")
        
    def _preload_models(self):
        """預載入模型池以避免運行時載入延遲"""
        logger.info("Preloading YOLO models...")
        
        def load_model(i):
            # 為每個預期的執行緒預載入模型
            dummy_thread_id = f"preload_{i}"
            logger.info(f"Loading {self.model_path} for TensorFlow Lite inference...")
            model = YOLO(self.model_path, task="detect")
            # 預熱模型
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            model.predict(dummy_input, verbose=False)
            return dummy_thread_id, model
        
        # 預載入最大工作者數量的模型
        with ThreadPoolExecutor(max_workers=self.max_workers) as preload_executor:
            futures = [preload_executor.submit(load_model, i) for i in range(self.max_workers)]
            for future in futures:
                thread_id, model = future.result()
                self.models[thread_id] = model
        
        logger.info(f"Preloaded {len(self.models)} YOLO models")
        
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
                    logger.debug(f"Reused preloaded model for thread {thread_id}")
                else:
                    # 如果沒有預載入模型，才創建新的
                    self.models[thread_id] = YOLO(self.model_path, task="detect")
                    logger.info(f"Created new model for thread {thread_id}")
        
        return self.models[thread_id]
    
    def predict_frame(self, frame_data):
        """在線程池中執行的推理函數"""
        frame, frame_id, worker_id = frame_data
        
        model = self.get_model_for_thread()
        
        start_time = time.time()
        result = model.predict(frame, verbose=False)[0]
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 記錄處理時間
        logger.info(f"PROCESSING_TIME,{frame_id},{worker_id},{processing_time:.4f}")
        
        return result, frame_id, processing_time

    async def producer(self, cap):
        """生產者：讀取視頻幀並放入隊列"""
        frame_id = 0
        
        while not self.should_stop:
            ret, frame = cap.read()
            if not ret:
                self.should_stop = True
                break
            
            frame_id += 1
            
            try:
                queue_size = self.frame_queue.qsize()
                # 記錄隊列長度
                logger.info(f"QUEUE_LENGTH,{frame_id},{queue_size}")
                await self.frame_queue.put((frame, frame_id))
                
            except asyncio.QueueFull:
                logger.info(f"QUEUE_FULL,{frame_id}")
                continue
                
            # 減少讓出頻率
            if frame_id % 5 == 0:
                await asyncio.sleep(0.001)
        
        # 發送結束信號
        for _ in range(self.max_workers):
            await self.frame_queue.put(None)
    
    async def worker(self, worker_id):
        """推理工作者 - 移除空閒超時退出邏輯"""
        consecutive_errors = 0
        
        try:
            while not self.should_stop:
                try:
                    frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=5.0)
                    
                    if frame_data is None:
                        break
                    
                    frame, frame_id = frame_data
                    consecutive_errors = 0
                    
                    loop = asyncio.get_event_loop()
                    result, result_frame_id, processing_time = await loop.run_in_executor(
                        self.executor, self.predict_frame, (frame, frame_id, worker_id)
                    )
                    
                    await self.result_queue.put((result, result_frame_id, frame, processing_time))
                    self.frame_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # 記錄超時
                    logger.info(f"WORKER_TIMEOUT,{worker_id}")
                    continue
                        
                except Exception as e:
                    consecutive_errors += 1
                    logger.info(f"WORKER_ERROR,{worker_id},{consecutive_errors},{str(e)}")
                    logger.error(f"Worker {worker_id} error: {str(e)}")
                    if consecutive_errors > 10:  # 增加容錯次數
                        logger.warning(f"Worker {worker_id} exiting due to consecutive errors")
                        break
                    continue
                    
        finally:
            async with self.workers_lock:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                    self.current_workers -= 1
                    logger.info(f"WORKER_TERMINATED,{worker_id},{self.current_workers}")
                    logger.info(f"Worker {worker_id} terminated, remaining workers: {self.current_workers}")
    
    async def workers_manager(self):
        """改進的動態工作者管理器 - 只增不減模式，無冷卻限制"""
        queue_stats = deque(maxlen=5)  # 減少統計窗口，更快響應
        
        while not self.should_stop:
            await asyncio.sleep(0.5)  # 更頻繁檢查
            
            current_queue_size = self.frame_queue.qsize()
            queue_stats.append(current_queue_size)
            
            if len(queue_stats) < 3:  # 減少最小統計數量
                continue
            
            avg_queue_length = sum(queue_stats) / len(queue_stats)
            
            # 記錄管理器狀態
            logger.info(f"MANAGER_STATUS,{self.current_workers},{avg_queue_length:.2f}")
            
            async with self.workers_lock:
                # 只增加工作者的邏輯 - 移除冷卻時間限制
                if avg_queue_length > 0.5 and self.current_workers < self.max_workers:
                    # 降低閾值到2.0，更積極地增加工作者
                    await self._add_worker()
                    logger.info(f"WORKER_ADDED,{self.current_workers}")
                    logger.info(f"Added worker, current workers: {self.current_workers}")
    
    async def _add_worker(self):
        """添加新的工作者"""
        self.worker_counter += 1
        worker_id = self.worker_counter
        
        worker_task = asyncio.create_task(self.worker(worker_id))
        self.workers[worker_id] = worker_task
        self.current_workers += 1
    
    async def consumer(self):
        """消費者：顯示推理結果"""
        processed_frames = {}
        next_display_frame = 1
        
        while not self.should_stop:
            try:
                result_data = await asyncio.wait_for(self.result_queue.get(), timeout=3.0)
                if result_data is None:
                    break
                
                result, frame_id, original_frame, processing_time = result_data
                processed_frames[frame_id] = (result, original_frame)
                
                # 記錄結果佇列狀態
                logger.info(f"RESULT_RECEIVED,{frame_id},{len(processed_frames)}")
                
                # 按順序顯示結果
                while next_display_frame in processed_frames:
                    display_result, display_frame = processed_frames.pop(next_display_frame)
                    
                    display_img = cv2.resize(display_result.plot(), (720, 480))
                    cv2.imshow("Adaptive YOLO Inference", display_img)
                    
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.should_stop = True
                        cv2.destroyAllWindows()
                        return
                    next_display_frame += 1
                self.result_queue.task_done()
            except asyncio.TimeoutError:
                logger.debug("Consumer timeout waiting for results")
                continue
            
        cv2.destroyAllWindows()

    async def run(self, video_path):
        """主要的推理流程"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return
        
        try:
            logger.info(f"Starting pipeline with video: {video_path}")
            logger.info(f"PIPELINE_STARTED,{video_path}")
            
            async with self.workers_lock:
                for i in range(self.min_workers):
                    await self._add_worker()
            
            tasks = [
                asyncio.create_task(self.producer(cap)),
                asyncio.create_task(self.consumer()),
                asyncio.create_task(self.workers_manager())
            ]
            
            async with self.workers_lock:
                for worker_task in self.workers.values():
                    tasks.append(worker_task)
            
            await asyncio.gather(*tasks[:3])
            
            async with self.workers_lock:
                remaining_workers = list(self.workers.values())
            
            if remaining_workers:
                await asyncio.gather(*remaining_workers, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
        finally:
            self.should_stop = True
            logger.info("Pipeline stopped")
            logger.info("PIPELINE_STOPPED")
            self.executor.shutdown(wait=True)
            cap.release()
            cv2.destroyAllWindows()

async def main():
    parser = argparse.ArgumentParser(description="自適應 YOLO 並發推理")
    parser.add_argument("--video_path", type=str, default="./data/video.mp4")
    parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
    parser.add_argument("--min_workers", type=int, default=1, help="最小工作者數量")
    parser.add_argument("--max_workers", type=int, default=8, help="最大工作者數量")
    parser.add_argument("--queue_size", type=int, default=32, help="幀隊列大小")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日誌級別")
    args = parser.parse_args()
    
    # 設置日誌級別
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    bin_dir = './bin'
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir)
    os.mkdir(bin_dir)
    
    pipeline = InferencePipeline(
        model_path=args.tflite_model,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        queue_size=args.queue_size
    )
    
    await pipeline.run(args.video_path)

if __name__ == "__main__":
    asyncio.run(main())
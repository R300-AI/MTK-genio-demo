from ultralytics import YOLO
import cv2
import os
import shutil
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from collections import deque
import numpy as np
import logging
import warnings
import sys
import signal

# 取消警告
warnings.filterwarnings("ignore")

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('performance_stats.txt', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger('ultralytics_demo')

class InferencePipeline:
    """高性能自適應 YOLO 並發推理管道"""
    
    def __init__(self, model_path, min_workers=1, max_workers=8, queue_size=10):
        # 基本配置
        self.model_path = model_path
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        # 線程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 隊列系統
        self.frame_queue = asyncio.Queue(maxsize=queue_size * 2)
        self.result_queue = asyncio.Queue(maxsize=queue_size)
        
        # 模型管理
        self.models = {}
        self.models_lock = threading.Lock()
        
        # 工作者管理
        self.workers = {}
        self.worker_counter = 0
        self.workers_lock = asyncio.Lock()
        
        # 控制標誌
        self.should_stop = False
        self.shutdown_timer = None
        
        # 初始化
        self._log_initialization()
        self._preload_models()
        
    def _log_initialization(self):
        """記錄初始化信息"""
        logger.info("=== Performance Stats Started ===")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Workers: {self.min_workers}-{self.max_workers}")
        
    def _preload_models(self):
        """預載入模型池"""
        logger.info("Preloading YOLO models...")
        
        def load_model(i):
            dummy_thread_id = f"preload_{i}"
            logger.info(f"Loading {self.model_path}...")
            model = YOLO(self.model_path, task="detect")
            # 預熱模型
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            model.predict(dummy_input, verbose=False)
            return dummy_thread_id, model
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as preload_executor:
            futures = [preload_executor.submit(load_model, i) for i in range(self.max_workers)]
            for future in futures:
                thread_id, model = future.result()
                self.models[thread_id] = model
        
        logger.info(f"Preloaded {len(self.models)} YOLO models")
        
    def get_model_for_thread(self):
        """為當前線程獲取模型實例"""
        thread_id = threading.current_thread().ident
        
        with self.models_lock:
            if thread_id not in self.models:
                # 嘗試重用預載入的模型
                preload_keys = [k for k in self.models.keys() if str(k).startswith("preload_")]
                if preload_keys:
                    preload_key = preload_keys[0]
                    self.models[thread_id] = self.models.pop(preload_key)
                    logger.debug(f"Reused preloaded model for thread {thread_id}")
                else:
                    self.models[thread_id] = YOLO(self.model_path, task="detect")
                    logger.info(f"Created new model for thread {thread_id}")
        
        return self.models[thread_id]
    
    def predict_frame(self, frame_data):
        """推理函數"""
        frame, frame_id, worker_id = frame_data
        model = self.get_model_for_thread()
        
        start_time = time.time()
        result = model.predict(frame, verbose=False)[0]
        processing_time = time.time() - start_time
        
        logger.info(f"PROCESSING_TIME,{frame_id},{worker_id},{processing_time:.4f}")
        return result, frame_id, processing_time
    
    def start_shutdown_timer(self, delay):
        """啟動強制關閉定時器"""
        def force_exit():
            logger.warning("Force shutdown timer triggered!")
            print("Program taking too long to exit, forcing shutdown...")
            cv2.destroyAllWindows()
            os._exit(1)
        
        if self.shutdown_timer is None:
            self.shutdown_timer = threading.Timer(delay, force_exit)
            self.shutdown_timer.daemon = True
            self.shutdown_timer.start()
            logger.info(f"Shutdown timer started: {delay} seconds")

    async def producer(self, cap):
        """生產者：讀取視頻幀並放入隊列"""
        frame_id = 0
        
        try:
            while not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video ended")
                    self.should_stop = True
                    self.start_shutdown_timer(5.0)
                    break
                
                frame_id += 1
                
                try:
                    queue_size = self.frame_queue.qsize()
                    logger.info(f"QUEUE_LENGTH,{frame_id},{queue_size}")
                    await asyncio.wait_for(self.frame_queue.put((frame, frame_id)), timeout=0.1)
                    
                except (asyncio.QueueFull, asyncio.TimeoutError):
                    logger.info(f"QUEUE_FULL,{frame_id}")
                    continue
                    
                if frame_id % 5 == 0:
                    await asyncio.sleep(0.001)
        
        finally:
            # 發送結束信號
            for _ in range(self.max_workers):
                try:
                    await asyncio.wait_for(self.frame_queue.put(None), timeout=0.1)
                except:
                    pass
            logger.info("Producer finished")
    
    async def worker(self, worker_id):
        """推理工作者"""
        consecutive_errors = 0
        
        try:
            while not self.should_stop:
                try:
                    frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                    
                    if frame_data is None:
                        break
                    
                    frame, frame_id = frame_data
                    consecutive_errors = 0
                    
                    loop = asyncio.get_event_loop()
                    result, result_frame_id, processing_time = await loop.run_in_executor(
                        self.executor, self.predict_frame, (frame, frame_id, worker_id)
                    )
                    
                    await self.result_queue.put((result, result_frame_id, frame, processing_time))
                    
                    try:
                        self.frame_queue.task_done()
                    except ValueError:
                        pass  # 忽略 task_done() 錯誤
                    
                except asyncio.TimeoutError:
                    logger.info(f"WORKER_TIMEOUT,{worker_id}")
                    if self.should_stop:
                        break
                    continue
                        
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Worker {worker_id} error: {str(e)}")
                    if consecutive_errors > 10:
                        logger.warning(f"Worker {worker_id} exiting due to errors")
                        break
                    continue
                    
        finally:
            async with self.workers_lock:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                    self.current_workers -= 1
                    logger.info(f"WORKER_TERMINATED,{worker_id},{self.current_workers}")
    
    async def workers_manager(self):
        """動態工作者管理器"""
        queue_stats = deque(maxlen=5)
        
        try:
            while not self.should_stop:
                await asyncio.sleep(0.1)
                
                if self.should_stop:
                    break
                
                current_queue_size = self.frame_queue.qsize()
                queue_stats.append(current_queue_size)
                
                if len(queue_stats) < 3:
                    continue
                
                avg_queue_length = sum(queue_stats) / len(queue_stats)
                logger.info(f"MANAGER_STATUS,{self.current_workers},{avg_queue_length:.2f}")
                
                async with self.workers_lock:
                    if avg_queue_length > 0.5 and self.current_workers < self.max_workers:
                        await self._add_worker()
                        logger.info(f"WORKER_ADDED,{self.current_workers}")
        finally:
            logger.info("Workers manager finished")
    
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
        
        try:
            while not self.should_stop:
                try:
                    result_data = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                    if result_data is None:
                        break
                    
                    result, frame_id, original_frame, processing_time = result_data
                    processed_frames[frame_id] = (result, original_frame)
                    
                    logger.info(f"RESULT_RECEIVED,{frame_id},{len(processed_frames)}")
                    
                    # 按順序顯示結果
                    while next_display_frame in processed_frames:
                        display_result, display_frame = processed_frames.pop(next_display_frame)
                        
                        display_img = cv2.resize(display_result.plot(), (720, 480))
                        cv2.imshow("Adaptive YOLO Inference", display_img)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord('q'):  # ESC 或 'q' 鍵
                            logger.info("User requested stop")
                            self.should_stop = True
                            self.start_shutdown_timer(3.0)
                            return
                        next_display_frame += 1
                    
                    try:
                        self.result_queue.task_done()
                    except ValueError:
                        pass  # 忽略 task_done() 錯誤
                        
                except asyncio.TimeoutError:
                    if self.should_stop:
                        break
                    continue
                    
        finally:
            cv2.destroyAllWindows()
            logger.info("Consumer finished")

    async def run(self, video_path):
        """主要的推理流程"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return
        
        try:
            logger.info(f"Starting pipeline with video: {video_path}")
            logger.info("PIPELINE_STARTED")
            
            # 初始化工作者
            async with self.workers_lock:
                for i in range(self.min_workers):
                    await self._add_worker()
            
            # 創建主要任務
            main_tasks = [
                asyncio.create_task(self.producer(cap)),
                asyncio.create_task(self.consumer()),
                asyncio.create_task(self.workers_manager())
            ]
            
            # 等待主要任務完成
            await asyncio.gather(*main_tasks, return_exceptions=True)
            
            # 等待所有工作者完成（設置超時）
            async with self.workers_lock:
                remaining_workers = list(self.workers.values())
            
            if remaining_workers:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*remaining_workers, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Workers didn't finish in time")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
        finally:
            await self._cleanup_resources(cap)

    async def _cleanup_resources(self, cap):
        """清理所有資源"""
        self.should_stop = True
        logger.info("PIPELINE_STOPPED")
        
        # 關閉 ThreadPoolExecutor
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logger.warning(f"Error shutting down executor: {e}")
        
        # 清理 OpenCV
        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
        cap.release()
        
        # 清理模型
        with self.models_lock:
            self.models.clear()
        
        # 取消定時器
        if self.shutdown_timer and self.shutdown_timer.is_alive():
            self.shutdown_timer.cancel()
            logger.info("Shutdown timer cancelled")
        
        logger.info("All resources cleaned up")

async def main():
    """主函數"""
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
    
    # 準備工作目錄
    bin_dir = './bin'
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir)
    os.mkdir(bin_dir)
    
    # 創建並運行推理管道
    pipeline = InferencePipeline(
        model_path=args.tflite_model,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        queue_size=args.queue_size
    )
    
    try:
        await pipeline.run(args.video_path)
    except Exception as e:
        logger.error(f"Main error: {e}")
    finally:
        logger.info("Main function completed")


def signal_handler(signum, frame):
    """信號處理器"""
    print(f"Received signal {signum}, forcing exit...")
    cv2.destroyAllWindows()
    os._exit(0)


if __name__ == "__main__":
    # 註冊信號處理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Program finished")
        
        # 檢查是否還有活躍線程，如果有則強制退出
        active_threads = threading.active_count()
        if active_threads > 1:
            print(f"Warning: {active_threads} threads still active, forcing exit...")
            os._exit(0)
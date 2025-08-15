"""
高性能自適應 YOLO 並發推理 DEMO
=====================================

這是一個展示如何使用 asyncio 和 ThreadPoolExecutor 
實現高效 YOLO 視頻推理的示範程式。

核心架構：
- Producer: 讀取視頻幀
- Workers: 並發推理處理  
- Consumer: 顯示結果
- Manager: 動態調整工作者數量
"""

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

# 取消警告訊息
warnings.filterwarnings("ignore")

# 設定日誌輸出
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
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
        
        # 初始化
        self._log_initialization()
        self._preload_models()
        
    def _log_initialization(self):
        """記錄初始化信息"""
        logger.info("=== YOLO 並發推理 DEMO 啟動 ===")
        logger.info(f"模型路徑: {self.model_path}")
        logger.info(f"工作者範圍: {self.min_workers}-{self.max_workers}")
        
    def _preload_models(self):
        """預載入模型池以提升推理速度"""
        logger.info("正在預載入 YOLO 模型...")
        
        def load_model(i):
            model = YOLO(self.model_path, task="detect")
            # 預熱模型
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            model.predict(dummy_input, verbose=False)
            return f"preload_{i}", model
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as preload_executor:
            futures = [preload_executor.submit(load_model, i) for i in range(self.max_workers)]
            for future in futures:
                thread_id, model = future.result()
                self.models[thread_id] = model
        
        logger.info(f"成功預載入 {len(self.models)} 個模型實例")
        
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
                else:
                    # 如果沒有預載入模型，創建新的
                    self.models[thread_id] = YOLO(self.model_path, task="detect")
        
        return self.models[thread_id]
    
    def predict_frame(self, frame_data):
        """執行 YOLO 推理"""
        frame, frame_id, worker_id = frame_data
        model = self.get_model_for_thread()
        
        start_time = time.time()
        result = model.predict(frame, verbose=False)[0]
        processing_time = time.time() - start_time
        
        logger.info(f"幀 {frame_id} 處理完成，耗時 {processing_time:.4f}s (工作者 {worker_id})")
        return result, frame_id, processing_time

    async def producer(self, cap):
        """生產者：讀取視頻幀並放入隊列"""
        frame_id = 0
        logger.info("生產者開始讀取視頻幀...")
        
        try:
            while not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    logger.info("視頻讀取完畢")
                    self.should_stop = True
                    break
                
                frame_id += 1
                
                try:
                    await asyncio.wait_for(self.frame_queue.put((frame, frame_id)), timeout=0.1)
                except (asyncio.QueueFull, asyncio.TimeoutError):
                    logger.warning(f"幀 {frame_id} 跳過（隊列已滿）")
                    continue
                    
                # 每 5 幀休息一下，避免過快讀取
                if frame_id % 5 == 0:
                    await asyncio.sleep(0.001)
        
        finally:
            # 發送結束信號給所有工作者
            for _ in range(self.max_workers):
                try:
                    await asyncio.wait_for(self.frame_queue.put(None), timeout=0.1)
                except:
                    pass
            logger.info("生產者結束")
    
    async def worker(self, worker_id):
        """推理工作者"""
        logger.info(f"工作者 {worker_id} 啟動")
        
        try:
            while not self.should_stop:
                try:
                    # 從隊列獲取幀數據
                    frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                    
                    if frame_data is None:  # 結束信號
                        break
                    
                    frame, frame_id = frame_data
                    
                    # 在線程池中執行推理
                    loop = asyncio.get_event_loop()
                    result, result_frame_id, processing_time = await loop.run_in_executor(
                        self.executor, self.predict_frame, (frame, frame_id, worker_id)
                    )
                    
                    # 將結果放入結果隊列
                    await self.result_queue.put((result, result_frame_id, frame, processing_time))
                    
                except asyncio.TimeoutError:
                    if self.should_stop:
                        break
                    continue
                        
                except Exception as e:
                    logger.error(f"工作者 {worker_id} 發生錯誤: {str(e)}")
                    continue
                    
        finally:
            # 從工作者列表中移除自己
            async with self.workers_lock:
                if worker_id in self.workers:
                    del self.workers[worker_id]
                    self.current_workers -= 1
                    logger.info(f"工作者 {worker_id} 結束 (剩餘 {self.current_workers} 個)")
    
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
                
                # 根據隊列長度動態調整工作者數量
                async with self.workers_lock:
                    if avg_queue_length > 0.5 and self.current_workers < self.max_workers:
                        await self._add_worker()
                        logger.info(f"新增工作者，目前總數: {self.current_workers}")
                        
        finally:
            logger.info("工作者管理器結束")
    
    async def _add_worker(self):
        """添加新的工作者"""
        self.worker_counter += 1
        worker_id = self.worker_counter
        
        worker_task = asyncio.create_task(self.worker(worker_id))
        self.workers[worker_id] = worker_task
        self.current_workers += 1
    
    async def consumer(self):
        """消費者：顯示推理結果"""
        frame_count = 0
        total_inference_time = 0
        display_interval = 30  # 每30幀計算一次平均
        
        try:
            while not self.should_stop:
                try:
                    # 獲取推理結果
                    result_data = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                    
                    if result_data is None:  # 結束信號
                        break
                    
                    result, frame_id, frame, processing_time = result_data
                    
                    # 繪製檢測結果
                    if result and hasattr(result, 'boxes') and len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = result.names[cls]
                            
                            # 繪製檢測框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'{class_name} {conf:.2f}', 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 計算性能統計
                    frame_count += 1
                    total_inference_time += processing_time
                    
                    if frame_count % display_interval == 0:
                        avg_inference_time = total_inference_time / display_interval
                        fps = display_interval / total_inference_time if total_inference_time > 0 else 0
                        logger.info(f"處理 {frame_count} 幀 | 平均推理時間: {avg_inference_time:.3f}s | FPS: {fps:.1f}")
                        total_inference_time = 0
                    
                    # 顯示影像
                    cv2.imshow('YOLO 檢測', frame)
                    
                    # 檢查退出條件
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' 鍵或 ESC 鍵
                        logger.info("檢測到退出鍵，準備結束程式")
                        self.should_stop = True
                        break
                    
                except asyncio.TimeoutError:
                    if self.should_stop:
                        break
                    continue
                        
                except Exception as e:
                    logger.error(f"消費者發生錯誤: {str(e)}")
                    continue
                    
        finally:
            logger.info(f"消費者結束，總共處理了 {frame_count} 幀")
            # 消費者負責清理自己的 OpenCV 視窗
            cv2.destroyAllWindows()
            # 確保視窗真的關閉
            for _ in range(5):
                cv2.waitKey(1)

    async def run(self, video_path):
        """主要的推理流程"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"無法開啟視頻: {video_path}")
            return
        
        try:
            logger.info(f"開始處理視頻: {video_path}")
            
            # 初始化工作者
            async with self.workers_lock:
                for i in range(self.min_workers):
                    await self._add_worker()
            
            # 創建並運行主要任務
            main_tasks = [
                asyncio.create_task(self.producer(cap)),
                asyncio.create_task(self.consumer()),
                asyncio.create_task(self.workers_manager())
            ]
            
            # 等待所有主要任務完成
            await asyncio.gather(*main_tasks, return_exceptions=True)
            
            # 等待剩餘工作者完成
            async with self.workers_lock:
                remaining_workers = list(self.workers.values())
            
            if remaining_workers:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*remaining_workers, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("部分工作者未能及時結束")
            
        except Exception as e:
            logger.error(f"流程發生錯誤: {str(e)}")
        finally:
            # 確保資源在這裡就被清理，而不是等到 main() 函數
            await self._cleanup_resources(cap)

    async def _cleanup_resources(self, cap):
        """清理所有資源"""
        self.should_stop = True
        logger.info("開始清理資源...")
        
        # 關閉線程池
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
            logger.info("線程池已關閉")
        except Exception as e:
            logger.warning(f"關閉線程池時發生錯誤: {e}")
        
        # 清理視頻捕獲器
        try:
            cap.release()
            logger.info("視頻捕獲器已釋放")
        except Exception as e:
            logger.warning(f"釋放視頻捕獲器時發生錯誤: {e}")
        
        # 清理模型
        try:
            with self.models_lock:
                self.models.clear()
            logger.info("模型快取已清理")
        except Exception as e:
            logger.warning(f"清理模型時發生錯誤: {e}")
        
        logger.info("資源清理完成")

async def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="YOLO 物件檢測演示程式")
    parser.add_argument("--video_path", type=str, default="./data/video.mp4", help="輸入視頻路徑")
    parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite", help="TensorFlow Lite 模型路徑")
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
        logger.error(f"程式執行錯誤: {e}")
    # 移除這裡的 finally，讓 pipeline.run() 自己負責清理


if __name__ == "__main__":
    asyncio.run(main())
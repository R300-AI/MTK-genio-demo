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
import gc

# 配置設定
warnings.filterwarnings("ignore")

# 日誌配置 - 僅輸出到檔案
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_inference.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class YOLOInferencePipeline:
    DISPLAY_SIZE = (720, 480)
    WINDOW_NAME = "YOLO 物件檢測"
    
    def __init__(self, model_path: str, min_workers: int = 1, max_workers: int = 8, queue_size: int = 10):
        # 基本配置
        self.model_path = model_path
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = 0
        
        # 執行器和隊列
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="YOLO-Worker")
        self.frame_queue = asyncio.Queue(maxsize=queue_size * 2)
        self.result_queue = asyncio.Queue(maxsize=queue_size)
        
        # 模型管理
        self.models = {}
        self.models_lock = threading.Lock()
        
        # 工作者管理
        self.workers = {}
        self.worker_counter = 0
        self.workers_lock = asyncio.Lock()
        
        # 控制狀態
        self.should_stop = False
        self.shutdown_timer = None
        
        # 初始化系統
        self._initialize_system()
        
    def _initialize_system(self) -> None:
        """初始化系統組件"""
        logger.info("=== YOLO 推理系統啟動 ===")
        logger.info(f"模型: {self.model_path}")
        logger.info(f"工作者範圍: {self.min_workers}-{self.max_workers}")
        self._preload_models()
        
    def load_model(self, index: int) -> tuple:
        """載入單一模型實例"""
        thread_id = f"preload_{index}"
        logger.debug(f"載入模型實例 {index}")
        
        # 模型載入+預熱
        model = YOLO(self.model_path, task="detect")
        model.predict(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8), verbose=False)
        return thread_id, model

    def _preload_models(self) -> None:
        """預載入模型池以提升推理效能"""
        logger.info("預載入 YOLO 模型...")
        # 平行載入模型
        with ThreadPoolExecutor(max_workers=self.max_workers) as preload_executor:
            futures = [preload_executor.submit(self.load_model, i) for i in range(self.max_workers)]
            
            for future in futures:
                thread_id, model = future.result()
                self.models[thread_id] = model
        
        logger.info(f"成功預載入 {len(self.models)} 個模型實例")
        
    def get_model_for_thread(self) -> YOLO:
        """獲取對應執行緒的 YOLO 模型"""
        thread_id = threading.current_thread().ident
        
        with self.models_lock:
            if thread_id not in self.models:
                # 優先重用預載入的模型
                preload_keys = [k for k in self.models.keys() if str(k).startswith("preload_")]
                if preload_keys:
                    preload_key = preload_keys[0]
                    self.models[thread_id] = self.models.pop(preload_key)
                    logger.debug(f"重用預載入模型 (執行緒 {thread_id})")
                else:
                    # 建立新模型實例
                    self.models[thread_id] = YOLO(self.model_path, task="detect")
                    logger.info(f"建立新模型實例 (執行緒 {thread_id})")
        
        return self.models[thread_id]
    
    def _predict(self, frame_data: tuple) -> tuple:
        """執行 YOLO 推理工作"""
        frame, frame_id, worker_id = frame_data
        model = self.get_model_for_thread()
        start_time = time.time()
        try:
            # 執行推理
            results = model.predict(frame, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                processing_time = time.time() - start_time
                return result, frame_id, processing_time
            else:
                logger.warning(f"幀 {frame_id} 推理返回空結果")
                return None, frame_id, time.time() - start_time
                
        except Exception as e:
            logger.error(f"幀 {frame_id} 推理錯誤: {e}")
            return None, frame_id, time.time() - start_time
    
    async def producer(self, cap: cv2.VideoCapture) -> None:
        frame_id = 0
        logger.info("生產者啟動 - 開始讀取視頻幀")
        
        try:
            while not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    logger.info("視頻讀取完畢，啟動關閉程序")
                    self.should_stop = True
                    break
                
                frame_id += 1
                
                try:
                    await asyncio.wait_for(
                        self.frame_queue.put((frame, frame_id)), 
                        timeout=0.1
                    )
                    
                    # 定期記錄隊列狀態
                    if frame_id % 30 == 0:
                        queue_size = self.frame_queue.qsize()
                        logger.debug(f"隊列狀態: {queue_size} 幀待處理")
                        
                except (asyncio.QueueFull, asyncio.TimeoutError):
                    logger.warning(f"幀 {frame_id} 跳過 (隊列已滿)")
                    continue
                    
                # 控制讀取速度
                if frame_id % 5 == 0:
                    await asyncio.sleep(0.001)
        
        finally:
            # 發送結束信號給所有工作者
            await self._send_stop_signals()
            logger.info("生產者結束")
            
    async def _send_stop_signals(self) -> None:
        """發送停止信號給所有工作者"""
        for _ in range(self.max_workers):
            try:
                await asyncio.wait_for(self.frame_queue.put(None), timeout=0.1)
            except asyncio.TimeoutError:
                continue
    
    async def worker(self, worker_id: int) -> None:
        """
        工作者 - 負責從隊列獲取原始影像幀，並返回 YOLO 模型的推理結果
        """
        logger.info(f"工作者 {worker_id} 啟動")
        consecutive_errors = 0
        max_errors = 10
        try:
            while not self.should_stop:
                try:
                    # 等待新任務
                    frame_data = await asyncio.wait_for(
                        self.frame_queue.get(), 
                        timeout=1.0
                    )
                    
                    if frame_data is None:  # 停止信號
                        break
                    
                    frame, frame_id = frame_data
                    consecutive_errors = 0
                    
                    # 執行推理
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, 
                        self._predict, 
                        (frame, frame_id, worker_id)
                    )
                    
                    # 提交結果 (不再需要傳遞原始幀，因為 result.plot() 包含所有需要的資訊)
                    await self.result_queue.put(result)
                    
                except asyncio.TimeoutError:
                    if self.should_stop:
                        break
                    continue   
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"工作者 {worker_id} 錯誤 ({consecutive_errors}/{max_errors}): {e}")
                    
                    if consecutive_errors >= max_errors:
                        logger.warning(f"工作者 {worker_id} 因連續錯誤退出")
                        break
                    
        finally:
            await self._remove_worker(worker_id)
            
    async def _remove_worker(self, worker_id: int) -> None:
        """從工作者清單中移除指定工作者"""
        async with self.workers_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                self.current_workers -= 1
                logger.info(f"工作者 {worker_id} 已移除 (剩餘: {self.current_workers})")
    
    async def workers_manager(self) -> None:
        """工作管理者 - 負責動態調整工作者數量"""
        queue_stats = deque(maxlen=5)
        check_interval = 0.1
        min_stats_count = 3
        load_threshold = 0.5
        
        logger.info("工作者管理器啟動")
        try:
            while not self.should_stop:
                await asyncio.sleep(check_interval)
                if self.should_stop:
                    break
                # 收集隊列統計
                current_queue_size = self.frame_queue.qsize()
                queue_stats.append(current_queue_size)
                
                if len(queue_stats) < min_stats_count:
                    continue
                
                # 計算平均負載
                avg_queue_length = sum(queue_stats) / len(queue_stats)
                
                # 動態調整工作者
                async with self.workers_lock:
                    if (avg_queue_length > load_threshold and 
                        self.current_workers < self.max_workers):
                        await self._add_worker()
                        logger.info(f"新增工作者 (總數: {self.current_workers})")
                        
        except Exception as e:
            logger.error(f"工作者管理器錯誤: {e}")
        finally:
            logger.info("工作者管理器結束")
    
    async def _add_worker(self) -> None:
        """新增工作者"""
        self.worker_counter += 1
        worker_id = self.worker_counter
        
        worker_task = asyncio.create_task(self.worker(worker_id))
        self.workers[worker_id] = worker_task
        self.current_workers += 1
    
    async def consumer(self) -> None:
        """
        結果消費者 - 負責從結果隊列獲取推理結果，使用 result.plot() 繪製並顯示結果
        """
        frame_count = 0
        total_processing_time = 0
        
        logger.info("消費者啟動 - 開始顯示結果")
        
        try:
            while not self.should_stop:
                try:
                    # 等待結果
                    result_data = await asyncio.wait_for(
                        self.result_queue.get(), 
                        timeout=1.0
                    )
                    
                    if result_data is None:
                        break
                    
                    result, frame_id, processing_time = result_data
                    
                    # 檢查推理結果是否有效
                    if result is None:
                        logger.warning(f"跳過幀 {frame_id} (推理結果為 None)")
                        continue
                    
                    # Consumer 負責繪製工作：使用 YOLO 內建的 plot() 方法
                    try:
                        plotted_img = result.plot()
                        if plotted_img is not None and plotted_img.size > 0:
                            # 調整大小並顯示
                            display_frame = cv2.resize(plotted_img, self.DISPLAY_SIZE)
                            cv2.imshow(self.WINDOW_NAME, display_frame)
                            
                            # 檢查使用者輸入
                            if key == 27 or key == ord('q'):
                                logger.info("使用者要求退出")
                                self.should_stop = True
                                break
                                
                            # 更新統計
                            frame_count += 1
                            total_processing_time += processing_time
                            avg_time = total_processing_time / frame_count
                            fps = frame_count / total_processing_time if total_processing_time > 0 else 0
                            logger.info(f"處理統計: {frame_count} 幀 {plotted_img.shape} {plotted_img.dtype}, 平均: {avg_time:.3f}s, FPS: {fps:.1f}")
                        else:
                            logger.warning(f"幀 {frame_id} 的 plot() 返回無效影像")
                            
                    except Exception as e:
                        logger.error(f"幀 {frame_id} 繪製錯誤: {e}")
                        continue

                except asyncio.TimeoutError:
                    if self.should_stop:
                        break
                    continue
                    
        finally:
            self._cleanup_display()
            logger.info(f"消費者結束 - 總共處理 {frame_count} 幀")
        
    def _cleanup_display(self) -> None:
        """清理顯示視窗"""
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    async def run(self, video_path: str) -> None:
        """主執行序 - 啟動生產者、消費者和工作者管理器"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"無法開啟視頻: {video_path}")
            raise ValueError(f"無法開啟視頻檔案: {video_path}")
        
        try:
            logger.info(f"開始處理視頻: {video_path}")
            
            # 啟動初始工作者
            await self._start_initial_workers()
            
            # 建立主要任務
            tasks = [
                asyncio.create_task(self.producer(cap), name="Producer"),
                asyncio.create_task(self.consumer(), name="Consumer"),
                asyncio.create_task(self.workers_manager(), name="WorkersManager")
            ]
            
            # 等待所有任務完成
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 等待剩餘工作者完成
            await self._wait_for_workers()
            
        except Exception as e:
            logger.error(f"推理管道錯誤: {e}")
            raise
        finally:
            await self._cleanup_resources(cap)
            
    async def _start_initial_workers(self) -> None:
        """啟動初始工作者"""
        async with self.workers_lock:
            for i in range(self.min_workers):
                await self._add_worker()
        logger.info(f"啟動了 {self.min_workers} 個初始工作者")
    
    async def _wait_for_workers(self) -> None:
        """等待所有工作者完成"""
        async with self.workers_lock:
            remaining_workers = list(self.workers.values())
        
        if remaining_workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining_workers, return_exceptions=True),
                    timeout=2.0
                )
                logger.info("所有工作者已完成")
            except asyncio.TimeoutError:
                logger.warning("部分工作者未及時完成")

    async def _cleanup_resources(self, cap: cv2.VideoCapture) -> None:
        """清理所有系統資源"""
        self.should_stop = True
        logger.info("開始清理系統資源")
        
        # 1. 取消剩餘的 asyncio 任務
        try:
            current_task = asyncio.current_task()
            all_tasks = [task for task in asyncio.all_tasks() if task != current_task]
            
            if all_tasks:
                for task in all_tasks:
                    task.cancel()
                
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*all_tasks, return_exceptions=True), 
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    pass
            logger.debug("✓ 取消asyncio任務")
        except Exception as e:
            logger.warning(f"✗ 取消asyncio任務: {e}")
        
        # 2. 關閉執行緒池
        try:
            if hasattr(self.executor, '_threads'):
                for thread in self.executor._threads:
                    if thread.is_alive():
                        thread.join(timeout=0.1)
            
            self.executor.shutdown(wait=False, cancel_futures=True)
            logger.debug("✓ 關閉執行緒池")
        except Exception as e:
            logger.warning(f"✗ 關閉執行緒池: {e}")
        
        # 3. 釋放視頻捕獲器
        try:
            if cap and cap.isOpened():
                cap.release()
            logger.debug("✓ 釋放視頻捕獲器")
        except Exception as e:
            logger.warning(f"✗ 釋放視頻捕獲器: {e}")
        
        # 4. 清理模型快取
        try:
            with self.models_lock:
                model_count = len(self.models)
                for model_key in list(self.models.keys()):
                    del self.models[model_key]
                self.models.clear()
                logger.debug(f"✓ 清理模型快取 ({model_count} 個實例)")
        except Exception as e:
            logger.warning(f"✗ 清理模型快取: {e}")
        
        # 5. 強制垃圾回收
        gc.collect()
        logger.info("資源清理完成")

async def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 高性能並發推理演示程式", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video_path", type=str, default="./data/video.mp4", help="輸入視頻檔案路徑")
    parser.add_argument("--model_path", type=str, default="./models/yolov8n_float32.tflite", help="YOLO 模型檔案路徑")
    parser.add_argument("--min_workers", type=int, default=1, help="最小工作者數量")
    parser.add_argument("--max_workers", type=int, default=8, help="最大工作者數量")
    parser.add_argument("--queue_size", type=int, default=32, help="處理隊列大小")
    args = parser.parse_args()
    
    # 設定日誌級別為 INFO（只輸出到檔案）
    logging.getLogger().setLevel(logging.INFO)

    pipeline = YOLOInferencePipeline(
        model_path=args.model_path,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        queue_size=args.queue_size
    )
    await pipeline.run(args.video_path)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程式被使用者中斷")
    except Exception as error:
        print(f"程式發生錯誤: {error}")
    finally:
        print("程式結束")

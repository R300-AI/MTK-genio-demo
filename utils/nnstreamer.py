#!/usr/bin/env python3
"""
NNStreamer 核心模組
包含 NNStreamer 主要類別和效能監控器
"""

import asyncio
import cv2
import numpy as np
import time
import threading
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import psutil
import gc

# 設定日誌
def setup_logging():
    """設定日誌配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nnstreamer.log', mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SlidingWindowController:
    """滑動窗口幀控制器 - 處理worker速度變化"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.recent_decisions = deque(maxlen=window_size)
        self.startup_frames = 0  # 啟動階段計數器
        self.startup_threshold = 30  # 啟動階段幀數
        
    def should_process_frame(self, queue_usage: float) -> bool:
        """基於隊列壓力動態決策是否處理幀"""
        self.startup_frames += 1
        
        # 啟動階段：處理所有幀，讓系統穩定
        if self.startup_frames <= self.startup_threshold:
            self.recent_decisions.append(1)
            return True
        
        # 更溫和的目標計算：避免過度激進
        # 只有當隊列使用率超過70%時才開始減少處理
        if queue_usage <= 0.7:
            target_keep_ratio = 1.0  # 正常處理所有幀
        else:
            # 當隊列使用率 > 70% 時，逐漸減少處理
            # 使用平方根函數使變化更平緩
            pressure = (queue_usage - 0.7) / 0.3  # 將70%-100%映射到0-1
            target_keep_ratio = max(0.3, 1.0 - (pressure ** 0.5) * 0.7)
        
        # 計算最近的實際處理比率
        if len(self.recent_decisions) == 0:
            current_ratio = 1.0  # 初始狀態：處理所有幀
        else:
            current_ratio = sum(self.recent_decisions) / len(self.recent_decisions)
        
        # 更保守的決策：只有當明顯超出目標時才減少處理
        should_process = current_ratio <= target_keep_ratio * 1.1  # 10%容忍度
        
        # 記錄決策到滑動窗口
        self.recent_decisions.append(1 if should_process else 0)
        
        return should_process

class PerformanceMonitor:
    """效能監控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.display_timestamps = deque(maxlen=window_size)  # 顯示時間戳
        self.cpu_readings = deque(maxlen=20)  # CPU 使用率移動窗口 (20個樣本)
        self.start_time = None  # 系統啟動時間
        self.total_displayed_frames = 0  # 實際顯示的幀數
        
    def add_inference_time(self, inference_time: float):
        """添加推論時間"""
        self.inference_times.append(inference_time)
        
    def add_display_sample(self):
        """添加顯示樣本 - 每次實際顯示幀時調用"""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            
        self.display_timestamps.append(current_time)
        self.total_displayed_frames += 1
        
        # 同時添加CPU使用率樣本
        self.cpu_readings.append(psutil.cpu_percent())
        
    def get_avg_inference_time(self) -> float:
        """獲取平均推論時間"""
        return np.mean(self.inference_times) if self.inference_times else 0.0
        
    def get_fps(self) -> float:
        """獲取實際顯示FPS"""
        if len(self.display_timestamps) < 2:
            return 0.0
            
        # 基於最近時間窗口計算
        recent_window = min(30, len(self.display_timestamps))  # 最近30幀
        if recent_window < 2:
            return 0.0
            
        time_span = self.display_timestamps[-1] - self.display_timestamps[-recent_window]
        if time_span <= 0:
            return 0.0
            
        return (recent_window - 1) / time_span
        
    def get_overall_fps(self) -> float:
        """獲取整體平均FPS"""
        if self.start_time is None or self.total_displayed_frames < 2:
            return 0.0
            
        elapsed_time = time.time() - self.start_time
        if elapsed_time <= 0:
            return 0.0
            
        return self.total_displayed_frames / elapsed_time
        
    def get_avg_cpu_percent(self) -> float:
        """獲取移動窗口平均CPU使用率"""
        if len(self.cpu_readings) == 0:
            return psutil.cpu_percent()
        return np.mean(self.cpu_readings)
    def get_system_stats(self) -> Dict[str, Any]:
        """獲取系統統計資訊"""
        return {
            'cpu_percent': self.get_avg_cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'avg_inference_ms': self.get_avg_inference_time() * 1000,
            'fps': self.get_fps(),
            'overall_fps': self.get_overall_fps()
        }


class NNStreamer:
    """
    NNStreamer 主要類別
    實現高效的模型串流推論架構
    """
    
    def __init__(self, 
                 interpreter_class,
                 model_path: str,
                 input_source: str = None,
                 max_workers: int = 4,
                 queue_size: int = 32,
                 display_output: bool = True,
                 enable_auto_tuning: bool = False):
        
        self.interpreter_class = interpreter_class
        self.model_path = model_path
        self.input_source = input_source or 0  # 預設使用攝影機
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.display_output = display_output
        self.enable_auto_tuning = enable_auto_tuning
        
        # 性能調優參數
        self.frame_timeout = 0.1  # 幀隊列超時時間
        self.result_timeout = 0.1  # 結果隊列超時時間
        self.frame_skip_interval = 1  # 幀跳過間隔
        self.frame_counter = 0  # 幀計數器
        
        # 統計計數器
        self.total_frames = 0
        self.dropped_frames = 0
        self.dropped_results = 0
        
        # 幀順序控制
        self.next_display_frame_id = 0  # 下一個應該顯示的幀ID
        self.pending_results = {}  # 等待顯示的結果 {frame_id: result}
        self.max_pending_frames = 10  # 最大等待幀數，防止記憶體累積
        
        # 滑動窗口控制器
        self.window_controller = SlidingWindowController(window_size=8)
        
        # 初始化組件
        self.interpreters = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AI-Worker")
        self.performance_monitor = PerformanceMonitor()
        
        # 異步隊列
        self.frame_queue = None
        self.result_queue = None
        
        # 控制狀態
        self.should_stop = False
        self.is_running = False
        
        # 初始化系統
        self._initialize_system()
        
    def _initialize_system(self):
        """初始化系統組件"""
        logger.info("=== NNStreamer 系統啟動 ===")
        logger.info(f"模型: {self.model_path}")
        logger.info(f"輸入來源: {self.input_source}")
        logger.info(f"最大工作者數: {self.max_workers}")
        
        # 預載入解釋器
        self._preload_interpreters()
        
    def _preload_interpreters(self):
        """預載入解釋器"""
        logger.info("預載入解釋器...")
        
        # 為每個工作者創建獨立的解釋器
        for i in range(self.max_workers):
            thread_id = f"worker_{i}"
            self.interpreters[thread_id] = self.interpreter_class(
                model_path=self.model_path
            )
            
        logger.info(f"成功預載入 {len(self.interpreters)} 個解釋器")
        
    def get_interpreter_for_thread(self):
        """獲取對應執行緒的解釋器"""
        thread_id = threading.current_thread().name
        
        # 如果當前執行緒沒有對應解釋器，創建新的
        if thread_id not in self.interpreters:
            worker_keys = [k for k in self.interpreters.keys() if k.startswith("worker_")]
            if worker_keys:
                # 重用預載入的解釋器
                available_key = worker_keys[0]
                self.interpreters[thread_id] = self.interpreters.pop(available_key)
            else:
                # 創建新解釋器
                self.interpreters[thread_id] = self.interpreter_class(
                    model_path=self.model_path
                )
                
        return self.interpreters[thread_id]
        
    async def frame_producer(self, source):
        """影像幀生產者協程"""
        logger.info(f"啟動影像擷取器: {source}")
        
        if isinstance(source, str) and source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            logger.error(f"無法開啟影像來源: {source}")
            return
            
        frame_count = 0
        try:
            while not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and source != "0":
                        logger.info("影片播放完畢")
                        break
                    else:
                        continue
                
                self.total_frames += 1
                
                # 使用滑動窗口控制器動態決策
                queue_usage = self.frame_queue.qsize() / self.queue_size
                
                if not self.window_controller.should_process_frame(queue_usage):
                    self.dropped_frames += 1
                    continue
                        
                # 將幀放入隊列
                try:
                    await asyncio.wait_for(
                        self.frame_queue.put((frame_count, frame)), 
                        timeout=self.frame_timeout
                    )
                    frame_count += 1
                except asyncio.TimeoutError:
                    self.dropped_frames += 1
                    drop_rate = (self.dropped_frames / self.total_frames) * 100
                    logger.warning(f"幀隊列已滿，跳過幀 (丟幀率: {drop_rate:.1f}%)")
                    
                    # 自動調整建議
                    if self.enable_auto_tuning and drop_rate > 10:
                        logger.info("🔧 調優建議: 考慮增加 --queue_size 參數或減少 --workers 數量")
                    continue
                    
        except Exception as e:
            logger.error(f"影像擷取錯誤: {e}")
        finally:
            cap.release()
            # 發送結束信號
            await self.frame_queue.put(None)
            logger.info("影像擷取器停止")
            
    def sync_inference(self, frame_data: Tuple[int, np.ndarray]) -> Optional[Tuple[int, np.ndarray, Any]]:
        """同步推論函數（在執行緒中運行）"""
        frame_id, frame = frame_data
        
        try:
            # 獲取解釋器
            interpreter = self.get_interpreter_for_thread()
            
            # 執行推論
            start_time = time.time()
            result = interpreter.inference(frame)
            
            # 記錄推論時間
            inference_time = time.time() - start_time
            self.performance_monitor.add_inference_time(inference_time)
            
            return frame_id, frame, result
            
        except Exception as e:
            logger.error(f"推論失敗 (Frame {frame_id}): {e}")
            return None
            
    async def inference_manager(self):
        """推論管理器協程"""
        logger.info("啟動推論管理器")
        
        active_tasks = set()
        
        try:
            while not self.should_stop:
                # 從幀隊列獲取數據
                frame_data = await self.frame_queue.get()
                
                if frame_data is None:
                    logger.info("收到結束信號")
                    break
                    
                # 創建推論任務
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(
                    self.executor, 
                    self.sync_inference, 
                    frame_data
                )
                active_tasks.add(task)
                
                # 處理完成的任務
                done_tasks = [t for t in active_tasks if t.done()]
                for task in done_tasks:
                    active_tasks.remove(task)
                    result = await task
                    
                    if result is not None:
                        try:
                            # 將結果放入隊列，但不立即處理順序
                            await asyncio.wait_for(
                                self.result_queue.put(result),
                                timeout=0.05
                            )
                        except asyncio.TimeoutError:
                            self.dropped_results += 1
                            logger.warning("結果隊列已滿，丟棄結果")
                            if self.enable_auto_tuning:
                                logger.info("🔧 調優建議: 考慮增加 --workers 數量或簡化 visualize() 方法")
                            
                # 限制同時進行的任務數量
                while len(active_tasks) >= self.max_workers:
                    done, active_tasks = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in done:
                        result = await task
                        if result is not None:
                            try:
                                await asyncio.wait_for(
                                    self.result_queue.put(result),
                                    timeout=self.result_timeout
                                )
                            except asyncio.TimeoutError:
                                self.dropped_results += 1
                                logger.warning("結果隊列已滿，丟棄結果")
                                if self.enable_auto_tuning:
                                    logger.info("🔧 調優建議: 考慮增加 --workers 數量或簡化 visualize() 方法")
                                    
        except Exception as e:
            logger.error(f"推論管理器錯誤: {e}")
        finally:
            # 等待所有任務完成
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            logger.info("推論管理器停止")
            
    async def result_consumer(self):
        """結果消費者協程 - 使用解釋器的視覺化方法，保證幀順序"""
        logger.info("啟動結果處理器")
        
        if self.display_output:
            cv2.namedWindow("NNStreamer Output", cv2.WINDOW_AUTOSIZE)
            
        # 獲取一個解釋器用於繪圖
        interpreter = list(self.interpreters.values())[0]
            
        try:
            while not self.should_stop:
                try:
                    # 獲取推論結果
                    result = await asyncio.wait_for(
                        self.result_queue.get(),
                        timeout=1.0
                    )
                    
                    if result is None:
                        break
                        
                    frame_id, frame, model_result = result
                    
                    # 將結果存入待處理字典
                    self.pending_results[frame_id] = (frame, model_result)
                    
                    # 按順序處理可以顯示的幀
                    while self.next_display_frame_id in self.pending_results:
                        display_frame, display_result = self.pending_results.pop(self.next_display_frame_id)
                        
                        # 更新效能監控 - 記錄實際顯示
                        self.performance_monitor.add_display_sample()
                        
                        # 使用解釋器的視覺化方法
                        if hasattr(interpreter, 'visualize'):
                            annotated_frame = interpreter.visualize(display_frame, display_result)
                        else:
                            # 回退到基本顯示
                            annotated_frame = display_frame
                        
                        # 獲取統計資訊
                        stats = self.performance_monitor.get_system_stats()
                        
                        # 顯示統計資訊
                        info_text = [
                            f"Frame: {self.next_display_frame_id}",
                            f"FPS: {stats['fps']:.1f}",
                            f"Overall FPS: {stats['overall_fps']:.1f}",
                            f"Inference: {stats['avg_inference_ms']:.1f}ms",
                            f"CPU: {stats['cpu_percent']:.1f}%",
                            f"Memory: {stats['memory_percent']:.1f}%",
                            f"Queue: {self.frame_queue.qsize()}/{self.queue_size}",
                            f"Drop Rate: {(self.dropped_frames / max(1, self.total_frames) * 100):.1f}%",
                            f"Pending: {len(self.pending_results)}",
                            f"Controller: {'Startup' if self.window_controller.startup_frames <= self.window_controller.startup_threshold else 'Active'}"
                        ]
                        
                        for i, text in enumerate(info_text):
                            cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 顯示影像
                        if self.display_output:
                            cv2.imshow("NNStreamer Output", cv2.resize(annotated_frame, (720, 480)))
                            
                            # 檢查按鍵退出
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q') or key == 27:  # 'q' 或 ESC
                                logger.info("使用者請求退出")
                                self.should_stop = True
                                break
                                
                        # 定期記錄統計資訊
                        if self.next_display_frame_id % 30 == 0:  # 每30幀記錄一次
                            stats = self.performance_monitor.get_system_stats()
                            
                            # 計算統計信息
                            drop_rate = (self.dropped_frames / max(1, self.total_frames)) * 100
                            result_drop_rate = (self.dropped_results / max(1, self.next_display_frame_id)) * 100
                            
                            logger.info(f"統計 - FPS: {stats['fps']:.1f}, "
                                      f"整體FPS: {stats['overall_fps']:.1f}, "
                                      f"推論時間: {stats['avg_inference_ms']:.1f}ms, "
                                      f"CPU: {stats['cpu_percent']:.1f}%, "
                                      f"記憶體: {stats['memory_percent']:.1f}%, "
                                      f"丟幀率: {drop_rate:.1f}%, "
                                      f"結果丟棄率: {result_drop_rate:.1f}%, "
                                      f"隊列使用率: {(self.frame_queue.qsize() / self.queue_size * 100):.1f}%, "
                                      f"等待幀數: {len(self.pending_results)}")
                            
                            # 自動調優建議
                            if self.enable_auto_tuning:
                                self._auto_tuning_suggestions(stats, drop_rate, result_drop_rate)
                        
                        # 移動到下一幀
                        self.next_display_frame_id += 1
                        
                        # 如果退出循環就停止處理
                        if self.should_stop:
                            break
                    
                    # 清理過舊的等待幀，防止記憶體累積
                    if len(self.pending_results) > self.max_pending_frames:
                        # 移除最舊的幀
                        oldest_frames = sorted(self.pending_results.keys())[:len(self.pending_results) - self.max_pending_frames]
                        for old_frame_id in oldest_frames:
                            self.pending_results.pop(old_frame_id, None)
                            # 如果移除的是應該顯示的幀，跳到下一個
                            if old_frame_id == self.next_display_frame_id:
                                self.next_display_frame_id = old_frame_id + 1
                        
                        logger.warning(f"清理過舊幀，跳到 frame {self.next_display_frame_id}")

                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error(f"結果處理器錯誤: {e}")
        finally:
            if self.display_output:
                cv2.destroyAllWindows()
            logger.info("結果處理器停止")
            
    async def run(self):
        """運行 NNStreamer 主循環"""
        if self.is_running:
            logger.warning("NNStreamer 已在運行中")
            return
            
        self.is_running = True
        self.should_stop = False
        
        # 初始化異步隊列
        self.frame_queue = asyncio.Queue(maxsize=self.queue_size)
        self.result_queue = asyncio.Queue(maxsize=self.queue_size)
        
        logger.info("=== 啟動 NNStreamer 串流處理 ===")
        
        try:
            # 創建所有協程任務
            tasks = [
                asyncio.create_task(self.frame_producer(self.input_source)),
                asyncio.create_task(self.inference_manager()),
                asyncio.create_task(self.result_consumer())
            ]
            
            # 並行運行所有任務
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("收到中斷信號")
        except Exception as e:
            logger.error(f"運行時錯誤: {e}")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """清理資源"""
        logger.info("清理系統資源...")
        self.should_stop = True
        self.is_running = False
        
        # 關閉執行器
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            
        # 清理解釋器
        self.interpreters.clear()
        
        # 強制垃圾回收
        gc.collect()
        
        logger.info("資源清理完成")
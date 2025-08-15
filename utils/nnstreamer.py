#!/usr/bin/env python3
import asyncio
import cv2
import numpy as np
import time
import threading
import logging
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

class AdaptiveDisplayController:
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.last_display_time = 0
        self.display_time_history = deque(maxlen=30)
        
    def should_display_now(self) -> tuple[bool, float]:
        current_time = time.time()
        
        if self.last_display_time == 0:
            self.last_display_time = current_time
            return True, 0
            
        elapsed_time = current_time - self.last_display_time
        
        if elapsed_time >= self.target_interval:
            self.display_time_history.append(elapsed_time)
            self.last_display_time = current_time
            return True, 0
        else:
            wait_time = self.target_interval - elapsed_time
            return False, wait_time



class SlidingWindowController:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.recent_decisions = deque(maxlen=window_size)
        self.startup_frame_count = 0
        self.startup_threshold = 30
        
    def should_process_frame(self, queue_usage_ratio: float) -> bool:
        self.startup_frame_count += 1
        
        if self.startup_frame_count <= self.startup_threshold:
            self.recent_decisions.append(1)
            return True
        
        if queue_usage_ratio <= 0.7:
            target_keep_ratio = 1.0
        else:
            pressure_factor = (queue_usage_ratio - 0.7) / 0.3
            target_keep_ratio = max(0.3, 1.0 - (pressure_factor ** 0.5) * 0.7)
        
        if len(self.recent_decisions) == 0:
            current_ratio = 1.0
        else:
            current_ratio = sum(self.recent_decisions) / len(self.recent_decisions)
        
        should_process = current_ratio <= target_keep_ratio * 1.1
        
        self.recent_decisions.append(1 if should_process else 0)
        
        return should_process

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_time_history = deque(maxlen=window_size)
        self.display_timestamp_history = deque(maxlen=window_size)
        self.cpu_usage_history = deque(maxlen=20)
        self.system_start_time = None
        self.total_displayed_frame_count = 0
        
    def add_inference_time(self, inference_duration: float):
        self.inference_time_history.append(inference_duration)
        
    def add_display_sample(self):
        current_timestamp = time.time()
        
        if self.system_start_time is None:
            self.system_start_time = current_timestamp
            
        self.display_timestamp_history.append(current_timestamp)
        self.total_displayed_frame_count += 1
        
        self.cpu_usage_history.append(psutil.cpu_percent())
        
    def get_avg_inference_time(self) -> float:
        return np.mean(self.inference_time_history) if self.inference_time_history else 0.0
        
    def get_fps(self) -> float:
        if len(self.display_timestamp_history) < 2:
            return 0.0
            
        recent_window_size = min(30, len(self.display_timestamp_history))
        if recent_window_size < 2:
            return 0.0
            
        time_span = self.display_timestamp_history[-1] - self.display_timestamp_history[-recent_window_size]
        if time_span <= 0:
            return 0.0
            
        return (recent_window_size - 1) / time_span
        
    def get_overall_fps(self) -> float:
        if self.system_start_time is None or self.total_displayed_frame_count < 2:
            return 0.0
            
        elapsed_time = time.time() - self.system_start_time
        if elapsed_time <= 0:
            return 0.0
            
        return self.total_displayed_frame_count / elapsed_time
        
    def get_avg_cpu_percent(self) -> float:
        if len(self.cpu_usage_history) == 0:
            return 0.0
        return np.mean(self.cpu_usage_history)
    
    def get_system_stats(self) -> Dict[str, Any]:
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
    
    參數說明:
    - target_fps: 目標幀率 (預設: None)
      * None: 自動設定FPS (影片檔案自動檢測，攝像頭使用30.0)
      * 指定數值: 強制使用該FPS，如 24.0, 30.0, 60.0 等
    
    使用範例:
    # 自動設定FPS (預設)
    neural_streamer = NNStreamer(executor, model_file_path, video_file_path)
    
    # 強制使用24 FPS播放
    neural_streamer = NNStreamer(executor, model_file_path, video_file_path, target_fps=24.0)
    
    # 強制使用60 FPS播放
    neural_streamer = NNStreamer(executor, model_file_path, video_file_path, target_fps=60.0)
    """
    
    def __init__(self, 
                 executor,
                 model_path: str,
                 input_source: str = None,
                 max_workers: int = 4,
                 max_queue_length: int = 32,
                 display_output: bool = True,
                 target_fps: float = None):
        
        self.executor = executor
        self.model_path = model_path
        self.input_source = input_source or 0
        self.max_worker_count = max_workers
        self.max_queue_length = max_queue_length
        self.display_output = display_output
        self.target_fps = target_fps
        
        self.frame_timeout = 0.1
        self.result_timeout = 0.1
        
        self.total_frame_count = 0
        self.dropped_frame_count = 0
        self.dropped_result_count = 0
        
        self.is_video_file = self._detect_video_file(input_source)
        
        self.next_display_frame_id = 0
        self.pending_result_dict = {}
        self.max_pending_frame_count = 10
        self.last_display_timestamp = time.time()
        self.frame_timeout_threshold = 2.0
        
        self.sliding_window_controller = SlidingWindowController(window_size=8)
        
        initial_fps = self.target_fps or 30.0
        self.display_controller = AdaptiveDisplayController(target_fps=initial_fps)
        
        self.executor_dict = {}
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_worker_count, thread_name_prefix="AI-Worker")
        self.performance_monitor = PerformanceMonitor()
        
        self.frame_queue = None
        self.result_queue = None
        
        self.should_stop = False
        self.is_running = False
        
        self.video_end_reached = False
        self.all_frames_processed = False
        self.total_video_frame_count = 0
        self.processed_video_frame_count = 0
        
        self.video_fps = 30.0
        self.frame_interval = 1.0 / self.video_fps
        self.last_frame_timestamp = 0
        
        self.active_worker_count = 0
        self._worker_lock = threading.Lock()
        
        # 初始化系統
        self._initialize_system()
        
    def _detect_video_file(self, input_source: str) -> bool:
        """檢測輸入源是否為影片文件"""
        if input_source is None:
            return False
        
        # 如果是數字字符串或單個數字，認為是攝像頭
        if isinstance(input_source, int) or (isinstance(input_source, str) and input_source.isdigit()):
            return False
            
        # 檢查文件副檔名
        if isinstance(input_source, str):
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
            return any(input_source.lower().endswith(ext) for ext in video_extensions)
        
        return False
        
    def _initialize_system(self):
        """初始化系統組件"""
        logger.info("============ NNStreamer System Initialization ============")
        logger.info(f"Model Path: {self.model_path}")
        logger.info(f"Input Source: {self.input_source}")
        logger.info(f"Input Type: {'Video File' if self.is_video_file else 'Camera/Live Stream'}")
        logger.info(f"Max Workers: {self.max_worker_count}")
        logger.info(f"Queue Size: {self.max_queue_length}")
        
        if self.target_fps is not None:
            logger.info(f"Target FPS: {self.target_fps:.2f}")
        else:
            logger.info("FPS Mode: Auto Detection (Video Files) / 30.0 FPS (Camera)")
        
        if self.is_video_file:
            logger.info("Processing Mode: Sequential Frame Processing (Video)")
        else:
            logger.info("Processing Mode: Adaptive Frame Skipping (Live Stream)")
        
        self._preload_executors()
        
    def _preload_executors(self):
        logger.info("Preloading AI Model Executors...")
        
        for i in range(self.max_worker_count):
            worker_thread_id = f"worker_{i}"
            self.executor_dict[worker_thread_id] = self.executor(
                model_path=self.model_path
            )
            
        logger.info(f"Successfully Preloaded {len(self.executor_dict)} Executors")
        
    def get_executor_for_thread(self):
        current_thread_id = threading.current_thread().name
        
        if current_thread_id not in self.executor_dict:
            available_worker_keys = [k for k in self.executor_dict.keys() if k.startswith("worker_")]
            if available_worker_keys:
                available_key = available_worker_keys[0]
                self.executor_dict[current_thread_id] = self.executor_dict.pop(available_key)
            else:
                self.executor_dict[current_thread_id] = self.executor(
                    model_path=self.model_path
                )
                
        return self.executor_dict[current_thread_id]

    async def frame_producer(self, input_source):
        logger.info(f"Starting Frame Capture System: {input_source}")
        
        if isinstance(input_source, str) and input_source.isdigit():
            video_capture = cv2.VideoCapture(int(input_source))
        else:
            video_capture = cv2.VideoCapture(input_source)
            
        if not video_capture.isOpened():
            logger.error(f"FAILED: Unable to Open Input Source: {input_source}")
            return
        
        if self.is_video_file:
            detected_fps = video_capture.get(cv2.CAP_PROP_FPS)
            
            if self.target_fps is not None:
                self.video_fps = self.target_fps
                fps_source = "Specified"
            elif detected_fps > 0:
                self.video_fps = detected_fps
                fps_source = "Auto"
            else:
                self.video_fps = 30.0
                fps_source = "Default"
            
            self.frame_interval = 1.0 / self.video_fps
            
            self.display_controller = AdaptiveDisplayController(target_fps=self.video_fps)
            
            logger.info(f"Video FPS: {self.video_fps:.2f} ({fps_source}), Frame Interval: {self.frame_interval:.4f}s")
            if detected_fps > 0 and detected_fps != self.video_fps:
                logger.info(f"Original Detected FPS: {detected_fps:.2f}")
        else:
            if self.target_fps is not None:
                self.video_fps = self.target_fps
                fps_source = "User Specified"
            else:
                self.video_fps = 30.0
                fps_source = "Auto Configuration"
            
            self.frame_interval = 1.0 / self.video_fps
            logger.info(f"Live Stream FPS: {self.video_fps:.2f} ({fps_source})")
            
        frame_counter = 0
        self.last_frame_timestamp = time.time()
        
        try:
            while not self.should_stop:
                frame_read_success, current_frame = video_capture.read()
                if not frame_read_success:
                    if isinstance(input_source, str) and input_source != "0":
                        logger.info("Video Playback Completed Successfully")
                        break
                    else:
                        continue
                
                self.total_frame_count += 1
                
                if self.frame_interval > 0:
                    current_timestamp = time.time()
                    expected_timestamp = self.last_frame_timestamp + self.frame_interval
                    
                    if current_timestamp < expected_timestamp:
                        wait_duration = expected_timestamp - current_timestamp
                        await asyncio.sleep(wait_duration)
                        current_timestamp = time.time()
                    
                    self.last_frame_timestamp = current_timestamp
                
                queue_usage_ratio = self.frame_queue.qsize() / self.max_queue_length
                
                if self.is_video_file:
                    should_process_current_frame = self._should_process_video_frame(queue_usage_ratio, frame_counter)
                else:
                    should_process_current_frame = self.sliding_window_controller.should_process_frame(queue_usage_ratio)
                
                if not should_process_current_frame:
                    self.dropped_frame_count += 1
                    continue
                        
                try:
                    if self.is_video_file:
                        await self.frame_queue.put((frame_counter, current_frame))
                        self.total_video_frame_count += 1
                    else:
                        await asyncio.wait_for(
                            self.frame_queue.put((frame_counter, current_frame)), 
                            timeout=self.frame_timeout
                        )
                    frame_counter += 1
                except asyncio.TimeoutError:
                    self.dropped_frame_count += 1
                    drop_rate = (self.dropped_frame_count / self.total_frame_count) * 100
                    logger.warning(f"WARNING: Frame Queue Full - Dropping Frame (Drop Rate: {drop_rate:.1f}%)")
                    continue
                    
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Frame Capture Failed: {e}")
        finally:
            video_capture.release()
            
            if self.is_video_file:
                self.video_end_reached = True
                logger.info(f"Video Reading Completed: {self.total_video_frame_count} Frames Sent to Processing Queue")
            
            await self.frame_queue.put(None)
            logger.info("Frame Capture System Stopped")
            
    def _should_process_video_frame(self, queue_usage: float, frame_count: int) -> bool:
        """影片幀處理策略 - 逐幀處理保持完整性"""
        # 影片文件逐幀處理，不跳幀，保持時間軸完整性
        # 參數保留供未來擴展使用
        return True
            
    def sync_inference(self, frame_data: Tuple[int, np.ndarray]) -> Optional[Tuple[int, np.ndarray, Any]]:
        """同步推論函數（在執行緒中運行）"""
        frame_id, input_frame = frame_data
        
        try:
            with self._worker_lock:
                self.active_worker_count += 1
            
            model_executor = self.get_executor_for_thread()
            
            start_timestamp = time.time()
            inference_result = model_executor.inference(input_frame)
            
            inference_duration = time.time() - start_timestamp
            self.performance_monitor.add_inference_time(inference_duration)
            
            return frame_id, input_frame, inference_result
            
        except Exception as error:
            logger.error(f"INFERENCE FAILED for Frame {frame_id}: {error}")
            return None
        finally:
            with self._worker_lock:
                self.active_worker_count -= 1
            
    async def inference_manager(self):
        """推論管理器協程 - 修復並行處理"""
        logger.info("Starting AI Inference Manager")
        
        active_inference_tasks = set()
        
        try:
            while not self.should_stop:
                # 非阻塞地清理已完成的任務
                completed_tasks = set()
                for task in list(active_inference_tasks):
                    if task.done():
                        completed_tasks.add(task)
                
                # 處理已完成的任務
                for completed_task in completed_tasks:
                    active_inference_tasks.remove(completed_task)
                    try:
                        task_result = await completed_task
                        if task_result is not None:
                            try:
                                await asyncio.wait_for(
                                    self.result_queue.put(task_result),
                                    timeout=0.05
                                )
                            except asyncio.TimeoutError:
                                self.dropped_result_count += 1
                                logger.warning("WARNING: Result Queue Full - Dropping Inference Result")
                    except Exception as e:
                        logger.error(f"任務執行錯誤: {e}")
                
                # 如果有容量，啟動新的推論任務
                if len(active_inference_tasks) < self.max_worker_count:
                    try:
                        frame_input_data = await asyncio.wait_for(
                            self.frame_queue.get(), 
                            timeout=0.01  # 短超時，避免阻塞
                        )
                        
                        if frame_input_data is None:
                            logger.info("Shutdown Signal Received - Stopping Inference Manager")
                            break
                        
                        # 立即啟動新的推論任務
                        event_loop = asyncio.get_event_loop()
                        inference_task = event_loop.run_in_executor(
                            self.thread_executor, 
                            self.sync_inference, 
                            frame_input_data
                        )
                        active_inference_tasks.add(inference_task)
                        
                    except asyncio.TimeoutError:
                        # 沒有新幀可處理，繼續下一個循環
                        pass
                else:
                    # 如果已達到最大並行數，稍微等待
                    await asyncio.sleep(0.001)
                
                # 給其他協程執行機會
                await asyncio.sleep(0.001)
                                    
        except Exception as error:
            logger.error(f"CRITICAL ERROR in Inference Manager: {error}")
        finally:
            if active_inference_tasks:
                logger.info(f"Waiting for {len(active_inference_tasks)} Inference Tasks to Complete...")
                task_results = await asyncio.gather(*active_inference_tasks, return_exceptions=True)
                
                for task_result in task_results:
                    if task_result is not None and not isinstance(task_result, Exception):
                        try:
                            await asyncio.wait_for(
                                self.result_queue.put(task_result),
                                timeout=2.0
                            )
                        except asyncio.TimeoutError:
                            self.dropped_result_count += 1
                            logger.warning("WARNING: Final Result Queue Full - Dropping Result")
                
                logger.info("All Inference Tasks Completed Successfully")
            
            await asyncio.sleep(0.1)
            await self.result_queue.put(None)
            logger.info("Inference Manager Stopped Successfully")
            
    async def result_consumer(self):
        """結果消費者協程 - 使用解釋器的視覺化方法，保證幀順序"""
        logger.info("Starting Result Consumer")
        
        if self.display_output:
            cv2.namedWindow("NNStreamer Output", cv2.WINDOW_AUTOSIZE)
            
        # 獲取一個執行器用於繪圖
        model_executor = list(self.executor_dict.values())[0]
            
        try:
            while not self.should_stop:
                try:
                    # 獲取推論結果
                    result = await asyncio.wait_for(
                        self.result_queue.get(),
                        timeout=1.0
                    )
                    
                    if result is None:
                        logger.info("收到結束信號")
                        # 影片模式：需要等待剩餘結果處理完畢
                        if self.is_video_file:
                            logger.info(f"Video Mode Shutdown: Waiting for {len(self.pending_result_dict)} Remaining Results")
                            timeout_counter = 0
                            max_timeout_cycles = 300
                            
                            while len(self.pending_result_dict) > 0 and timeout_counter < max_timeout_cycles:
                                frames_processed_this_cycle = False
                                
                                while self.next_display_frame_id in self.pending_result_dict:
                                    display_frame, display_result = self.pending_result_dict.pop(self.next_display_frame_id)
                                    frames_processed_this_cycle = True
                                    self.last_display_timestamp = time.time()
                                    
                                    self.performance_monitor.add_display_sample()
                                    
                                    if hasattr(model_executor, 'visualize'):
                                        annotated_display_frame = model_executor.visualize(display_frame, display_result)
                                    else:
                                        annotated_display_frame = display_frame
                                    
                                    performance_stats = self.performance_monitor.get_system_stats()
                                    performance_info_text = [
                                        f"Frame: {self.next_display_frame_id}",
                                        f"FPS: {performance_stats['fps']:.1f}",
                                        f"Inference: {performance_stats['avg_inference_ms']:.1f}ms",
                                        f"CPU: {performance_stats['cpu_percent']:.1f}%",
                                        f"Memory: {performance_stats['memory_percent']:.1f}%",
                                        f"Workers: {self.active_worker_count}/{self.max_worker_count}"
                                    ]
                                    
                                    for text_index, display_text in enumerate(performance_info_text):
                                        cv2.putText(annotated_display_frame, display_text, (10, 30 + text_index * 25),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    if self.display_output:
                                        cv2.imshow("NNStreamer Output", annotated_display_frame)
                                        cv2.waitKey(1)
                                    
                                    self.next_display_frame_id += 1
                                
                                if not frames_processed_this_cycle:
                                    await asyncio.sleep(0.05)
                                    timeout_counter += 1
                                else:
                                    timeout_counter = 0
                            
                            if self.processed_video_frame_count >= self.total_video_frame_count and len(self.pending_result_dict) == 0:
                                self.all_frames_processed = True
                                logger.info(f"✅ VIDEO PROCESSING COMPLETED: {self.processed_video_frame_count} Frames Processed")
                                logger.info("All frames processed successfully - System shutdown initiated")
                            else:
                                logger.warning(f"⚠️ VIDEO PROCESSING INCOMPLETE: {self.processed_video_frame_count}/{self.total_video_frame_count} Frames, {len(self.pending_result_dict)} Results Pending")
                        
                        break
                        
                    current_frame_id, current_frame, current_model_result = result
                    
                    self.pending_result_dict[current_frame_id] = (current_frame, current_model_result)
                    
                    if self.is_video_file:
                        self.processed_video_frame_count += 1
                    
                    frames_processed_this_cycle = False
                    while self.next_display_frame_id in self.pending_result_dict:
                        display_frame, display_result = self.pending_result_dict.pop(self.next_display_frame_id)
                        frames_processed_this_cycle = True
                        self.last_display_timestamp = time.time()
                        
                        self.performance_monitor.add_display_sample()
                        
                        if hasattr(model_executor, 'visualize'):
                            annotated_display_frame = model_executor.visualize(display_frame, display_result)
                        else:
                            annotated_display_frame = display_frame
                        
                        performance_stats = self.performance_monitor.get_system_stats()

                        # 顯示統計資訊
                        info_text = [
                            f"Frame: {self.next_display_frame_id}",
                            f"FPS: {performance_stats['fps']:.1f}",
                            f"Inference: {performance_stats['avg_inference_ms']:.1f}ms",
                            f"CPU: {performance_stats['cpu_percent']:.1f}%",
                            f"Memory: {performance_stats['memory_percent']:.1f}%",
                            f"Workers: {self.active_worker_count}/{self.max_worker_count}"
                        ]
                        
                        for i, text in enumerate(info_text):
                            cv2.putText(annotated_display_frame, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
                        
                        # 顯示影像
                        if self.display_output:
                            cv2.imshow("NNStreamer Output", annotated_display_frame)
                            
                            # 影片模式：使用固定等待時間維持幀率
                            if self.is_video_file:
                                # 計算適合的等待時間（毫秒）
                                wait_ms = max(1, int(self.frame_interval * 1000))
                                key = cv2.waitKey(wait_ms) & 0xFF
                            else:
                                # 即時模式：使用最小等待時間
                                key = cv2.waitKey(1) & 0xFF
                            
                            if key == ord('q') or key == 27:  # 'q' 或 ESC
                                logger.info("使用者請求退出")
                                self.should_stop = True
                                break
                                
                        # 定期記錄統計資訊
                        if self.next_display_frame_id % 30 == 0:  # 每30幀記錄一次
                            stats = self.performance_monitor.get_system_stats()
                            
                            # 計算統計信息
                            drop_rate = (self.dropped_frame_count / max(1, self.total_frame_count)) * 100
                            result_drop_rate = (self.dropped_result_count / max(1, self.next_display_frame_id)) * 100
                            
                            logger.info(f"統計 - FPS: {stats['fps']:.1f}, "
                                      f"整體FPS: {stats['overall_fps']:.1f}, "
                                      f"推論時間: {stats['avg_inference_ms']:.1f}ms, "
                                      f"CPU: {stats['cpu_percent']:.1f}%, "
                                      f"記憶體: {stats['memory_percent']:.1f}%, "
                                      f"丟幀率: {drop_rate:.1f}%, "
                                      f"結果丟棄率: {result_drop_rate:.1f}%, "
                                      f"隊列使用率: {(self.frame_queue.qsize() / self.max_queue_length * 100):.1f}%, "
                                      f"等待幀數: {len(self.pending_result_dict)}")

                        # 移動到下一幀
                        self.next_display_frame_id += 1
                        
                        # 如果退出循環就停止處理
                        if self.should_stop:
                            break
                    
                    # 影片模式：檢查是否所有幀都已處理完畢
                    if self.is_video_file and self.video_end_reached:
                        if (self.processed_video_frame_count >= self.total_video_frame_count and 
                            len(self.pending_result_dict) == 0):
                            self.all_frames_processed = True
                            logger.info(f"影片處理完畢！總共處理了 {self.processed_video_frame_count} 幀")
                            logger.info("所有影像都已處理完成，準備關閉系統")
                            break
                    
                    # 超時保護：只在即時模式下啟用
                    if not self.is_video_file:
                        current_time = time.time()
                        if not frames_processed_this_cycle and (current_time - self.last_display_timestamp) > self.frame_timeout_threshold:
                            if len(self.pending_result_dict) > 0:
                                # 找到最小的可用幀ID
                                min_available_frame = min(self.pending_result_dict.keys())
                                if min_available_frame > self.next_display_frame_id:
                                    skipped_frames = min_available_frame - self.next_display_frame_id
                                    logger.warning(f"超時保護: 跳過 {skipped_frames} 幀 (從 {self.next_display_frame_id} 到 {min_available_frame - 1})")
                                    self.next_display_frame_id = min_available_frame
                                    self.last_display_timestamp = current_time
                    
                    # 清理過舊的等待幀，只在即時模式下啟用（影片模式需要所有幀）
                    if not self.is_video_file and len(self.pending_result_dict) > self.max_pending_frame_count:
                        # 保留最近的幀，移除距離當前顯示幀太遠的幀
                        sorted_frames = sorted(self.pending_result_dict.keys())
                        frames_to_remove = []
                        
                        for frame_id in sorted_frames[:-self.max_pending_frame_count]:
                            frames_to_remove.append(frame_id)
                        
                        for frame_id in frames_to_remove:
                            self.pending_result_dict.pop(frame_id, None)
                        
                        if frames_to_remove:
                            logger.warning(f"清理 {len(frames_to_remove)} 個過舊幀: {frames_to_remove[0]} 到 {frames_to_remove[-1]}")

                except asyncio.TimeoutError:
                    # 影片模式：如果沒有更多結果且影片已讀取完畢，檢查是否完成
                    if self.is_video_file and self.video_end_reached:
                        # 給更多時間等待最後的推論結果
                        if (self.processed_video_frame_count >= self.total_video_frame_count and 
                            len(self.pending_result_dict) == 0):
                            self.all_frames_processed = True
                            logger.info(f"影片處理完畢！總共處理了 {self.processed_video_frame_count} 幀")
                            logger.info("所有影像都已處理完成，準備關閉系統")
                            break
                        else:
                            # 如果還有未處理的幀，繼續等待
                            missing_frames = self.total_video_frame_count - self.processed_video_frame_count
                            logger.info(f"影片結束但仍有 {missing_frames} 幀未處理，{len(self.pending_result_dict)} 幀等待顯示，繼續等待...")
                            continue
                    
                    # 在超時時也檢查是否需要跳過等待的幀（主要針對即時模式）
                    if not self.is_video_file:
                        current_time = time.time()
                        if (current_time - self.last_display_timestamp) > self.frame_timeout_threshold:
                            if len(self.pending_result_dict) > 0:
                                min_available_frame = min(self.pending_result_dict.keys())
                                if min_available_frame > self.next_display_frame_id:
                                    skipped_frames = min_available_frame - self.next_display_frame_id
                                    logger.warning(f"超時跳幀: 跳過 {skipped_frames} 幀 (從 {self.next_display_frame_id} 到 {min_available_frame - 1})")
                                    self.next_display_frame_id = min_available_frame
                                    self.last_display_timestamp = current_time
                    continue
                    
        except Exception as e:
            logger.error(f"CRITICAL ERROR in Result Consumer: {e}")
        finally:
            if self.display_output:
                cv2.destroyAllWindows()
            
            if self.is_video_file and self.all_frames_processed:
                logger.info("✅ VIDEO PROCESSING FULLY COMPLETED - All Frames Processed and Displayed")
            
            logger.info("Result Consumer Stopped Successfully")

    async def run(self):
        """運行 NNStreamer 主循環"""
        if self.is_running:
            logger.warning("WARNING: NNStreamer Already Running")
            return
            
        self.is_running = True
        self.should_stop = False
        
        # 初始化異步隊列
        self.frame_queue = asyncio.Queue(maxsize=self.max_queue_length)
        self.result_queue = asyncio.Queue(maxsize=self.max_queue_length)
        
        logger.info("============ Starting NNStreamer Processing ============")
        
        try:
            # 創建所有協程任務
            tasks = [
                asyncio.create_task(self.frame_producer(self.input_source)),
                asyncio.create_task(self.inference_manager()),
                asyncio.create_task(self.result_consumer())
            ]
            
            # 並行運行所有任務
            await asyncio.gather(*tasks, return_exceptions=True)
            
            if self.is_video_file:
                logger.info("=== Video Processing Status Check ===")
                logger.info(f"Video Reading Complete: {self.video_end_reached}")
                logger.info(f"Total Frames Sent: {self.total_video_frame_count}")
                logger.info(f"Frames Processed: {self.processed_video_frame_count}")
                logger.info(f"Pending Results: {len(self.pending_result_dict) if hasattr(self, 'pending_result_dict') else 0}")
                logger.info(f"All Frames Processed: {self.all_frames_processed}")
                
                if self.all_frames_processed:
                    logger.info("🎬 VIDEO PROCESSING SUCCESSFULLY COMPLETED - All Frames Processed and Displayed")
                else:
                    logger.warning("⚠️ Video processing may be incomplete")
            
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt Received")
        except Exception as e:
            logger.error(f"Runtime Error: {e}")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """清理資源"""
        logger.info("Cleaning Up System Resources...")
        self.should_stop = True
        self.is_running = False
        
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=True)
            
        self.executor_dict.clear()
        
        gc.collect()
        
        logger.info("Resource Cleanup Completed")

import threading
import time
import cv2
import logging
from collections import deque
import psutil

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

class Monitor:
    def __init__(self, log_interval=10, window_size=8):
        logger.info(f"[MONITOR] log interval set to {log_interval} frames")
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # 各種計數器
        self.frame_count = 0        # Producer 產生的幀數
        self.processed_count = 0    # 已處理完成的幀數
        self.consumed_count = 0     # Consumer 顯示的幀數
        self.processing = 0         # 正在處理中的幀數
        
        # 統一的記錄頻率設定
        self.log_interval = log_interval
        #self.last_log_time = time.time()
        
        # 記錄上次觸發 log 的計數器值（用於判斷是否需要記錄）
        self.last_logged_produced = 0
        self.last_logged_processed = 0
        self.last_logged_consumed = 0

        self.produced_times = deque(maxlen=window_size)
        self.processed_times = deque(maxlen=window_size)
        self.consumed_times = deque(maxlen=window_size)

        # 跡象型 log: queue 停留時間追蹤
        self.input_queue_times = deque(maxlen=100)
        self.output_queue_times = deque(maxlen=100)
        self.last_queue_warning = 0
        self.last_queue_size = 0
        
        # FPS 計算結果初始值
        self.produced_fps = None
        self.processed_fps = None
        self.consumed_fps = None


    def draw_info(self, frame):
        with self.lock:
            info = f"FPS: {self.processed_fps}  Processors: {self.processing}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return frame

    def fps(self, times):
        # 只有當 deque 填滿 window_size 時才計算 FPS
        if len(times) < self.produced_times.maxlen:
            return None
        duration = times[-1] - times[0]
        # 如果時間間隔太短（小於0.1秒），認為數據不可靠
        if duration <= 0.1:
            return None
        return f"{(len(times) - 1) / duration:.2f}"

    def _check_and_log(self, event_type):
        """標準的 logging 檢查函數 - 只負責檢查是否需要記錄"""
        now = time.time()
        
        # 只記錄對應事件的時間，並更新對應的FPS
        if event_type == "produced":
            self.produced_times.append(now)
            self.produced_fps = self.fps(self.produced_times)
            # 跡象型 log: input queue 停留時間
            self.input_queue_times.append(now)
        elif event_type == "processed":
            self.processed_times.append(now)
            self.processed_fps = self.fps(self.processed_times)
            # 跡象型 log: output queue 停留時間
            self.output_queue_times.append(now)
        elif event_type == "consumed":
            self.consumed_times.append(now)
            self.consumed_fps = self.fps(self.consumed_times)
        
        # 只在有新的FPS計算時才print（避免重複打印）
        if event_type in ["produced", "processed", "consumed"]:
            # FPS 計算已完成，不需要額外處理
            pass

        should_log = False
        if event_type == "produced":
            if self.frame_count % self.log_interval == 0 and self.frame_count > self.last_logged_produced:
                should_log = True
                self.last_logged_produced = self.frame_count
        
        elif event_type == "processed":
            if self.processed_count % self.log_interval == 0 and self.processed_count > self.last_logged_processed:
                should_log = True
                self.last_logged_processed = self.processed_count
                
        elif event_type == "consumed":
            if self.consumed_count % self.log_interval == 0 and self.consumed_count > self.last_logged_consumed:
                should_log = True
                self.last_logged_consumed = self.consumed_count
        
        elif event_type == "processing_start":
            if self.processing % 5 == 0 and self.processing > 0:
                should_log = True
        
        elif event_type == "processing_end":
            if self.processing == 0 and self.frame_count > 0:
                should_log = True
        
        if should_log:
            queued = max(0, self.frame_count - self.processed_count - self.processing)
            pending = max(0, self.processed_count - self.consumed_count)
            status_msg = (
                f"[Monitor] -> "
                f"Produced: {self.frame_count}, "
                f"Queued: {queued}, "
                f"Processing: {self.processing}, "
                f"Processed: {self.processed_count}, "
                f"Pending: {pending}, "
                f"Consumed: {self.consumed_count}"
                f" | {event_type}"
            )
            logger.info(status_msg)

            # 跡象型 log: queue 停留時間統計
            if len(self.input_queue_times) > 1:
                avg_input_interval = (self.input_queue_times[-1] - self.input_queue_times[0]) / (len(self.input_queue_times) - 1)
                logger.debug(f"[DEBUG] [MONITOR] Avg input queue frame interval: {avg_input_interval:.4f}s")
            if len(self.output_queue_times) > 1:
                avg_output_interval = (self.output_queue_times[-1] - self.output_queue_times[0]) / (len(self.output_queue_times) - 1)
                logger.debug(f"[DEBUG] [MONITOR] Avg output queue frame interval: {avg_output_interval:.4f}s")

            # 跡象型 log: FPS 變化
            logger.debug(f"[DEBUG] [MONITOR] FPS: produced={self.produced_fps}, processed={self.processed_fps}, consumed={self.consumed_fps}")

            # 跡象型 log: queue 長時間未下降警告
            now = time.time()
            if queued == self.last_queue_size and queued > 0 and now - self.last_queue_warning > 5:
                logger.warning(f"[MONITOR] Queue size stagnant at {queued} for over 5s")
                self.last_queue_warning = now
            self.last_queue_size = queued
            
            # 添加系統資源使用率DEBUG日誌
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            if cpu_percent > 80 or memory_percent > 80:
                logger.debug(f"[DEBUG] [MONITOR] System resources: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%")
    
    # === 計數器更新方法 ===
    def count_produced(self):
        """Producer 產生新幀時調用"""
        with self.lock:
            self.frame_count += 1
            self._check_and_log("produced")
    
    def count_processing_start(self):
        """開始處理新幀時調用"""
        with self.lock:
            self.processing += 1
            self._check_and_log("processing_start")
    
    def count_processing_end(self):
        """完成處理幀時調用"""
        with self.lock:
            self.processing = max(0, self.processing - 1)
            self._check_and_log("processing_end")
    
    def count_processed(self):
        """完成處理幀時調用"""
        with self.lock:
            self.processed_count += 1
            self._check_and_log("processed")
    
    def count_consumed(self):
        """Consumer 顯示幀時調用"""
        with self.lock:
            self.consumed_count += 1
            self._check_and_log("consumed")
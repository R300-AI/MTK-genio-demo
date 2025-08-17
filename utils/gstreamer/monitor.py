import threading
import time
import cv2
import logging
from collections import deque

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

class Monitor:
    def __init__(self, log_interval=10, window_size=50):
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
        
        # FPS 計算結果初始值
        self.produced_fps = None
        self.processed_fps = None
        self.consumed_fps = None

    def draw_info(self, frame):
        with self.lock:
            info = f"FPS: {self.consumed_fps} Active Workers: {self.processing}"        
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return frame

    def fps(self, times):
        # 需要至少3個時間點來計算穩定的FPS（2個間隔）
        if len(times) < 3:
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
        elif event_type == "processed":
            self.processed_times.append(now)
            self.processed_fps = self.fps(self.processed_times)
        elif event_type == "consumed":
            self.consumed_times.append(now)
            self.consumed_fps = self.fps(self.consumed_times)
        
        # 只在有新的FPS計算時才print（避免重複打印）
        if event_type in ["produced", "processed", "consumed"]:
            print(
                f"Produced (fps: {self.produced_fps}), "
                f"Processed (fps: {self.processed_fps}), "
                f"Consumed (fps: {self.consumed_fps}), "
            )

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
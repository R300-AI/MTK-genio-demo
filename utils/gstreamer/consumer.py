import cv2
import logging
import time
from collections import deque

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

class Consumer:
    def __init__(self, window_name="", monitor=None, display_size=None):
        logger.info(f"[CONSUMER] window name - {window_name}")
        self.window_name = window_name
        self.monitor = monitor
        self.display_size = display_size
        self.frame_count = 0
        
        # Consumer 自己的 FPS 計算
        self.display_times = deque(maxlen=30)  # 保留最近30幀的顯示時間
        self.current_fps = 0.0

    def consume(self, frame):
        current_time = time.time()
        self.frame_count += 1
        
        # 記錄顯示時間並計算實際 FPS
        self.display_times.append(current_time)
        self._calculate_display_fps()
        
        try:
            if self.display_size is not None:
                frame = cv2.resize(frame, self.display_size)
            
            # 在幀上顯示實際的 Consumer FPS
            self._draw_fps_info(frame)

            cv2.imshow(self.window_name, frame)
            
            if self.monitor:
                self.monitor.count_consumed()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise StopIteration
                
        except Exception as e:
            logger.error(f"CONSUMER_CONSUME: Error displaying frame {self.frame_count}: {e}")

    def _calculate_display_fps(self):
        """計算實際的視覺更新 FPS"""
        if len(self.display_times) < 2:
            self.current_fps = 0.0
            return
            
        # 使用最近的時間計算 FPS
        time_span = self.display_times[-1] - self.display_times[0]
        if time_span > 0:
            frame_count = len(self.display_times) - 1
            self.current_fps = frame_count / time_span
        else:
            self.current_fps = 0.0
    
    def _draw_fps_info(self, frame):
        """在幀上繪製 FPS 資訊"""
        # 顯示實際的 Consumer FPS
        fps_text = f"Display FPS: {self.current_fps:.1f}"
        
        # 如果有 monitor，也顯示 active workers 資訊
        if self.monitor:
            with self.monitor.lock:
                active_workers = self.monitor.processing
            fps_text += f" | Active Workers: {active_workers}"
        
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

import cv2
import logging
import time
from collections import deque

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

import threading

class Consumer:
    def __init__(self, window_name="", monitor=None, display_size=None, fps=30, mode='camera', producer=None):
        logger.info(f"[CONSUMER] window name - {window_name}, mode={mode}")
        self.window_name = window_name
        self.monitor = monitor
        self.display_size = display_size
        self.fps = fps
        self.frame_count = 0
        self.display_times = deque(maxlen=30)
        self.current_fps = 0.0
        self.mode = mode
        self.producer = producer  # 新增：用於獲取原始video fps
        if mode == 'video':
            self.display_buffer = deque(maxlen=50)
        else:
            self.display_buffer = deque(maxlen=1)
        self.display_buffer_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._thread = None
        
        # 添加fps追蹤變數
        self.display_counter = 0
        self.last_display_fps_time = time.time()
        self.fps_check_interval = 30  # 每30幀檢查一次實際顯示fps
        self.buffer_size_log_counter = 0  # buffer size記錄計數器

    def start_display(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._display_loop, daemon=True)
            self._thread.start()

    def stop_display(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def put_frame(self, frame):
        with self.display_buffer_lock:
            if self.mode == 'video':
                # video mode: 確保不丟幀，適當等待 buffer 空間
                while len(self.display_buffer) >= self.display_buffer.maxlen:
                    time.sleep(0.001)  # 短暫等待，避免丟幀
            else:
                # camera mode: 保持原有邏輯，允許丟幀
                if len(self.display_buffer) == self.display_buffer.maxlen:
                    logger.warning(f"[CONSUMER] Display buffer full, dropping oldest frame.")
            self.display_buffer.append(frame)

    def _display_loop(self):
        # 動態設定顯示間隔：video mode使用原始fps，camera mode使用設定fps
        if self.mode == 'video' and self.producer and hasattr(self.producer, 'get_fps'):
            target_fps = self.producer.get_fps()
            interval = 1.0 / target_fps
            logger.info(f"[CONSUMER] Video mode: using original fps={target_fps:.2f}, interval={interval:.4f}s")
        else:
            interval = 1.0 / self.fps
            logger.info(f"[CONSUMER] Camera mode: using fixed fps={self.fps}, interval={interval:.4f}s")
        
        last_frame = None
        frames_without_new = 0  # 計算連續沒有新幀的次數
        
        while self._running.is_set():
            time.sleep(interval)
            frame = None
            
            with self.display_buffer_lock:
                if self.mode == 'video':
                    # video mode: 依序顯示所有 frame
                    if self.display_buffer:
                        frame = self.display_buffer.popleft()
                        frames_without_new = 0
                    else:
                        frames_without_new += 1
                else:
                    # camera mode: 只顯示最新 frame
                    while self.display_buffer:
                        frame = self.display_buffer.popleft()
                        frames_without_new = 0
            
            # 選擇要顯示的幀
            display_frame = frame if frame is not None else last_frame
            
            # Video mode: 如果連續超過 30 幀(1秒)都沒有新幀，停止重複顯示
            if self.mode == 'video' and frames_without_new > 30:
                display_frame = None
            
            if display_frame is not None:
                self.display_counter += 1
                try:
                    disp_frame = display_frame
                    if self.display_size is not None:
                        disp_frame = cv2.resize(disp_frame, self.display_size)
                    if self.monitor:
                        self.monitor.draw_info(disp_frame)
                    cv2.imshow(self.window_name, disp_frame)
                    
                    # Video mode: 只在顯示新幀時才計數，避免重複計數
                    if self.monitor and (frame is not None or self.mode == 'camera'):
                        self.monitor.count_consumed()
                        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_display()
                        break
                        
                    # 定期記錄顯示fps
                    if self.display_counter % self.fps_check_interval == 0:
                        current_time = time.time()
                        actual_interval = (current_time - self.last_display_fps_time) / self.fps_check_interval
                        actual_display_fps = 1.0 / actual_interval if actual_interval > 0 else 0
                        
                        logger.info(f"Consumer Display#{self.display_counter}, "
                                   f"Target_FPS={1.0/interval:.2f}, Actual_Display_FPS={actual_display_fps:.2f}")
                        self.last_display_fps_time = current_time
                        
                except Exception as e:
                    logger.error(f"CONSUMER_CONSUME: Error displaying frame {self.frame_count}: {e}")
            
            # 更新 last_frame 只在有新幀時
            if frame is not None:
                last_frame = frame
        
        # 清理CV2窗口
        cv2.destroyWindow(self.window_name)
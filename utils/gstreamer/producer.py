import cv2
import logging
import time

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

class Producer:
    def __init__(self, source, filename = None, index = None, monitor=None, balancer=None):
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self.mode = "camera"
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            self.mode = "video"
            self.cap = cv2.VideoCapture(source)

        logger.info(f"[PRODUCER] {self.mode} source - {source}")
        self.monitor = monitor
        self.balancer = balancer  # 新增 Balancer 支援

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        # 記錄視頻資訊
        target_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 設定播放控制參數
        if self.mode == "video":
            self.last_frame_time = 0
            self.frame_interval = 1.0 / target_fps
        else:
            self.frame_interval = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.balancer:
            self.frame_interval = self.balancer.get_producer_sleep(self.frame_interval)
    
        if self.mode == "video":
            elapsed = time.time() - self.last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
            self.last_frame_time = time.time()
        else:  # camera 模式
            time.sleep(self.frame_interval)
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        if self.monitor:
            self.monitor.count_produced()
        return frame

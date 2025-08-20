import cv2
import logging
import time

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

class Producer:
    def __init__(self, source, filename=None, index=None, monitor=None, mode=None):
        logger.info("=" * 60)
        logger.info("PRODUCER INITIALIZATION STARTED")
        logger.info("-" * 60)
        # ========================================
        # 模式判斷與基礎設定
        # ========================================
        self._determine_mode(mode, source)
        self.monitor = monitor
        
        # ========================================
        # 初始化VideoCapture (兩種模式共通)
        # ========================================
        self._initialize_capture(source)
        
        # ========================================
        # 設定模式特定參數
        # ========================================
        if self.mode == "camera":
            self._setup_camera_mode()
        else:
            self._setup_video_mode()
            
        # ========================================
        # 共通的追蹤變數設定
        # ========================================
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps_check_interval = 30  # 每30幀檢查一次實際fps

        logger.info(f" - Initial FPS: {self.target_fps}")
        logger.info(f" - Image Size: {self.width}x{self.height}")
        logger.info(f" -  Streaming Mode: {self.is_live_stream}")
        logger.info("=" * 60)

    def _determine_mode(self, mode, source):
        """決定運行模式：camera 或 video"""
        if mode is not None:
            self.mode = mode
        elif isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self.mode = "camera"
        else:
            self.mode = "video"
        logger.info(f"[PRODUCER] Mode determined: {self.mode}")

    def _initialize_capture(self, source):
        """初始化OpenCV VideoCapture物件"""
        if self.mode == "camera":
            self.cap = cv2.VideoCapture(int(source))
        else:
            self.cap = cv2.VideoCapture(source)
        logger.info(f" - Video source - {source}")

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {self.mode} source: {source}")

    def _setup_camera_mode(self):
        """Camera mode 特定設定"""
        # 獲取攝像頭參數
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.is_live_stream = True
        self.total_frames = -1  # 無限流

    def _setup_video_mode(self):
        """Video mode 特定設定"""
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video模式特定參數
        self.is_live_stream = False
        duration = self.total_frames / self.target_fps if self.target_fps > 0 else 0
        logger.info(f" - Video duration: {duration:.2f} seconds")

    def get_fps(self):
        """獲取視頻的原始 FPS"""
        return self.target_fps if self.target_fps > 0 else 30

    def get_total_frames(self):
        """獲取總幀數 (僅Video mode有意義)"""
        return self.total_frames if self.mode == "video" else -1

    def get_progress(self):
        """獲取播放進度 (僅Video mode有意義)"""
        if self.mode == "video" and self.total_frames > 0:
            return (self.frame_counter / self.total_frames) * 100
        return 0

    def __iter__(self):
        return self

    def __next__(self):
        frame_start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration

        # ========================================
        # 計算FPS
        # ========================================
        self.frame_counter += 1
        self._FPS_handler(frame_start_time)
        return frame

    def _FPS_handler(self, frame_start_time):
        """Camera mode 特定的幀處理邏輯"""
        # 定期記錄camera實際產生fps的DEBUG資訊
        if self.frame_counter % self.fps_check_interval == 0:
            current_time = time.time()
            actual_interval = (current_time - self.last_fps_time) / self.fps_check_interval
            actual_fps = 1.0 / actual_interval if actual_interval > 0 else 0
            frame_read_time = time.time() - frame_start_time
            logger.debug(f"[PRODUCER] Frame#{self.frame_counter}, Actual_FPS={actual_fps:.2f}, Frame_read_time={frame_read_time:.4f}")
            self.last_fps_time = current_time

    def cleanup(self):
        """釋放資源"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            if self.mode == "camera":
                logger.info("Camera capture released")
            else:
                logger.info("Video capture released")

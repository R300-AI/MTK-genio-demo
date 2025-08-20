"""
================================================================================
🎬 Producer 繼承架構設計
================================================================================

Producer類採用繼承架構，將通用邏輯抽象到基類，具體模式在子類中實現。
系統支援Video模式（完整性優先）和Camera模式（實時性優先）。

🎯 核心組件：
┌─────────────┬──────────────────┬─────────────────────────────────────────┐
│ 📸 Video    │ 文件幀生產       │ 有限幀數、進度追蹤、完整性保證          │
│ 📷 Camera   │ 實時幀生產       │ 無限流、自動重連、實時性優化            │
│ ⚙️ Config   │ 配置管理         │ 類型安全、模式特定、彈性配置            │
└─────────────┴──────────────────┴─────────────────────────────────────────┘

📊 資料流向：Source ──[VideoCapture]──> BaseProducer ──> Pipeline

🎯 繼承關係：
                    BaseProducer (抽象基類)
                    ├── Template Method Pattern
                    ├── 統一初始化流程
                    ├── 通用監控介面
                    └── 抽象方法定義
                           │
            ┌──────────────┴──────────────┐
            │                             │
    VideoProducer                  CameraProducer
    (完整性優先)                   (實時性優先)

📊 職責分配：
┌─────────────────┬──────────────────┬─────────────────┬─────────────────┐
│   功能類別      │   BaseProducer   │  VideoProducer  │ CameraProducer  │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ 🚀 初始化管理   │ ✅ 模板方法流程   │ ✅ 文件驗證     │ ✅ 攝像頭偵測   │
│ 🎯 Capture設置  │ ✅ 通用設置       │ ✅ 讀取優化     │ ✅ 緩衝區配置   │
│ 📊 參數配置     │ ✅ 基礎參數       │ ✅ 進度追蹤     │ ✅ 實時參數     │
│ 🎬 幀生產邏輯   │ 🔹 抽象方法       │ ✅ 順序讀取     │ ✅ 實時捕獲     │
│ 📈 性能監控     │ ✅ FPS追蹤       │ ✅ 進度報告     │ ✅ 延遲監控     │
│ 🔄 錯誤處理     │ ✅ 基礎處理       │ ✅ 文件錯誤     │ ✅ 重連機制     │
│ 🧹 資源清理     │ ✅ 統一清理       │ 🔹 繼承使用     │ 🔹 繼承使用     │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┘

🔧 核心特性：
• Template Method：統一初始化流程，子類實現特化邏輯
• 配置管理：ProducerConfig支援類型安全的參數配置
• 錯誤分層：ProducerException體系提供精確異常處理
• 工廠模式：create_producer()自動選擇適當的Producer類型

🛠️ 使用方式：
• 工廠創建：producer = create_producer(source, config)
• 直接創建：producer = VideoProducer(source, config)
• 向後相容：producer = Producer(source)  # 自動轉發到工廠函數
"""

import cv2
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generator, Tuple

logger = logging.getLogger('gstreamer_demo')

# ============================================================================
# 🔧 配置類定義
# ============================================================================

@dataclass
class ProducerConfig:
    """Producer通用配置類"""
    # FPS監控配置
    fps_check_interval: int = 30
    
    # Capture配置
    buffer_size: int = 1
    timeout: float = 5.0
    
    # 錯誤處理配置
    retry_count: int = 3
    reconnect_delay: float = 1.0

# ============================================================================
# ⚠️ 異常類定義
# ============================================================================

class ProducerException(Exception):
    """Producer基礎異常"""
    pass

class CaptureInitializationError(ProducerException):
    """Capture初始化失敗異常"""
    pass

class FrameReadError(ProducerException):
    """幀讀取失敗異常"""
    pass

class CameraConnectionError(ProducerException):
    """Camera連接錯誤異常"""
    pass

# ============================================================================
# 🏗️ BaseProducer 抽象基類
# ============================================================================

class BaseProducer(ABC):
    """Producer抽象基類"""
    
    def __init__(self, source, config: Optional[ProducerConfig] = None, monitor=None):
        self.source = source
        self.config = config or ProducerConfig()
        self.monitor = monitor
        
        # 通用追蹤變數
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.cap = None
        
        logger.info("=" * 60)
        logger.info(f"{self.__class__.__name__.upper()} INITIALIZATION STARTED")
        logger.info("-" * 60)
        
        # Template Method Pattern
        try:
            self._initialize_capture()
            self._configure_parameters()
            self._setup_monitoring()
            self._log_initialization_summary()
        except Exception as e:
            logger.error(f"Producer initialization failed: {e}")
            raise CaptureInitializationError(f"Failed to initialize {self.__class__.__name__}: {e}")
        
        logger.info("=" * 60)
    
    @property
    @abstractmethod
    def mode(self) -> str:
        """返回Producer模式標識"""
        pass
    
    @abstractmethod
    def _initialize_capture(self):
        """初始化VideoCapture物件"""
        pass
    
    @abstractmethod
    def _configure_parameters(self):
        """配置模式特定參數"""
        pass
    
    @abstractmethod
    def _get_next_frame(self):
        """獲取下一幀"""
        pass
    
    def _setup_monitoring(self):
        """設置監控系統"""
        if self.monitor:
            self.monitor.set_producer_info(
                mode=self.mode,
                total_frames=getattr(self, 'total_frames', -1),
                fps=getattr(self, 'target_fps', 30)
            )
            logger.debug("Monitor integration completed")
    
    def _log_initialization_summary(self):
        """記錄初始化摘要"""
        logger.info(f" - Mode: {self.mode}")
        logger.info(f" - Source: {self.source}")
        logger.info(f" - Target FPS: {getattr(self, 'target_fps', 'Unknown')}")
        logger.info(f" - Resolution: {getattr(self, 'width', 'Unknown')}x{getattr(self, 'height', 'Unknown')}")
        logger.info(f" - Live Stream: {getattr(self, 'is_live_stream', 'Unknown')}")
        if hasattr(self, 'total_frames') and self.total_frames > 0:
            duration = self.total_frames / getattr(self, 'target_fps', 30)
            logger.info(f" - Duration: {duration:.2f}s ({self.total_frames} frames)")
    
    def __iter__(self):
        logger.debug(f"[{self.mode.upper()}] Starting frame iteration")
        return self
    
    def __next__(self):
        try:
            return self._get_next_frame()
        except StopIteration:
            logger.info(f"[{self.mode.upper()}] Frame iteration completed. Total frames: {self.frame_counter}")
            self.cleanup()
            raise
        except Exception as e:
            logger.error(f"[{self.mode.upper()}] Frame read error: {e}")
            raise FrameReadError(f"Failed to read frame: {e}")
    
    def cleanup(self):
        """釋放資源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info(f"[{self.mode.upper()}] Capture resource released")
    
    def get_fps(self) -> float:
        return getattr(self, 'target_fps', 30.0)
    
    def get_total_frames(self) -> int:
        return getattr(self, 'total_frames', -1)

# ============================================================================
# 🎬 VideoProducer 實現類
# ============================================================================

class VideoProducer(BaseProducer):
    """Video文件Producer實現"""
    
    @property
    def mode(self) -> str:
        return "video"
    
    def _initialize_capture(self):
        logger.debug(f"[VIDEO] Initializing capture for: {self.source}")
        
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {self.source}")
            
            logger.info(f" - Video file loaded: {self.source}")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.debug("Video capture optimization applied")
            
        except Exception as e:
            logger.error(f"[VIDEO] Capture initialization failed: {e}")
            raise
    
    def _configure_parameters(self):
        logger.debug("[VIDEO] Configuring video-specific parameters")
        
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.is_live_stream = False
        self.frame_duration = 1.0 / self.target_fps if self.target_fps > 0 else 0.033
        
        logger.debug(f"[VIDEO] Parameters configured - FPS: {self.target_fps}, "
                    f"Resolution: {self.width}x{self.height}, Frames: {self.total_frames}")
    
    def _get_next_frame(self):
        frame_start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            logger.info(f"[VIDEO] Playback completed. Total frames: {self.frame_counter}")
            raise StopIteration
        
        self.frame_counter += 1
        self._handle_fps_monitoring(frame_start_time)
        
        return frame
    
    def _handle_fps_monitoring(self, frame_start_time):
        if self.frame_counter % self.config.fps_check_interval == 0:
            current_time = time.time()
            
            interval = (current_time - self.last_fps_time) / self.config.fps_check_interval
            actual_fps = 1.0 / interval if interval > 0 else 0
            frame_time = time.time() - frame_start_time
            progress = (self.frame_counter / self.total_frames) * 100 if self.total_frames > 0 else 0
            
            logger.debug(f"[VIDEO] Frame#{self.frame_counter}/{self.total_frames}, "
                        f"Progress={progress:.1f}%, FPS={actual_fps:.2f}, "
                        f"Frame_time={frame_time:.4f}s")
            
            self.last_fps_time = current_time
    
    def get_progress(self) -> float:
        if self.total_frames > 0:
            return (self.frame_counter / self.total_frames) * 100
        return 0.0

# ============================================================================
# 📷 CameraProducer 實現類
# ============================================================================

class CameraProducer(BaseProducer):
    """Camera實時Producer實現"""
    
    @property
    def mode(self) -> str:
        return "camera"
    
    def _initialize_capture(self):
        camera_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        logger.debug(f"[CAMERA] Initializing capture for camera ID: {camera_id}")
        
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera: {camera_id}")
            
            logger.info(f" - Camera connected: ID {camera_id}")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.debug(f"Camera buffer size set to: {self.config.buffer_size}")
            
        except Exception as e:
            logger.error(f"[CAMERA] Capture initialization failed: {e}")
            raise CameraConnectionError(f"Failed to connect to camera {camera_id}: {e}")
    
    def _configure_parameters(self):
        logger.debug("[CAMERA] Configuring camera-specific parameters")
        
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.is_live_stream = True
        self.total_frames = -1
        self.connection_lost_count = 0
        
        logger.debug(f"[CAMERA] Parameters configured - FPS: {self.target_fps}, "
                    f"Resolution: {self.width}x{self.height}")
    
    def _get_next_frame(self):
        frame_start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("[CAMERA] Frame read failed, attempting reconnection...")
            if self._attempt_reconnection():
                ret, frame = self.cap.read()
            
            if not ret:
                logger.error("[CAMERA] Camera connection lost permanently")
                raise CameraConnectionError("Camera connection lost")
        
        self.frame_counter += 1
        self._handle_fps_monitoring(frame_start_time)
        
        return frame
    
    def _handle_fps_monitoring(self, frame_start_time):
        if self.frame_counter % self.config.fps_check_interval == 0:
            current_time = time.time()
            
            interval = (current_time - self.last_fps_time) / self.config.fps_check_interval
            actual_fps = 1.0 / interval if interval > 0 else 0
            frame_time = time.time() - frame_start_time
            
            logger.debug(f"[CAMERA] Frame#{self.frame_counter}, "
                        f"FPS={actual_fps:.2f}, Frame_time={frame_time:.4f}s, "
                        f"Reconnects={self.connection_lost_count}")
            
            self.last_fps_time = current_time
    
    def _attempt_reconnection(self) -> bool:
        logger.info(f"[CAMERA] Connection lost, attempting reconnection...")
        
        for attempt in range(self.config.retry_count):
            logger.info(f"[CAMERA] Reconnection attempt {attempt + 1}/{self.config.retry_count}")
            
            if self.cap:
                self.cap.release()
            
            time.sleep(self.config.reconnect_delay)
            
            try:
                camera_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
                self.cap = cv2.VideoCapture(camera_id)
                
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                    self.connection_lost_count += 1
                    logger.info(f"[CAMERA] Reconnection successful on attempt {attempt + 1}")
                    return True
                    
            except Exception as e:
                logger.warning(f"[CAMERA] Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error("[CAMERA] All reconnection attempts failed")
        return False

# ============================================================================
# 🏭 工廠函數
# ============================================================================

def create_producer(source, config: Optional[ProducerConfig] = None, monitor=None) -> BaseProducer:
    """Producer工廠函數"""
    logger.info(f"[FACTORY] Creating producer for source: {source}")
    
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        producer = CameraProducer(source, config, monitor)
        logger.info("[FACTORY] Created CameraProducer for real-time processing")
    else:
        producer = VideoProducer(source, config, monitor)
        logger.info("[FACTORY] Created VideoProducer for file processing")
    
    return producer

# ============================================================================
# 🔄 向後相容性
# ============================================================================

class Producer(BaseProducer):
    def __new__(cls, source, filename=None, index=None, monitor=None, mode=None):
        logger.debug("[COMPATIBILITY] Using legacy Producer constructor, forwarding to factory")
        config = ProducerConfig()
        return create_producer(source, config, monitor)
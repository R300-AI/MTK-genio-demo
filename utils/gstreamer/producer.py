"""
================================================================================
🎬 Producer 架構設計
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
    (完整性優先)                     (實時性優先)

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
from utils.gstreamer.config import ProducerConfig

# ============================================================================
# ⚠️ 異常類定義
# ============================================================================
from utils.gstreamer.exceptions import CaptureInitializationError, FrameReadError, CameraConnectionError

# ============================================================================
# 🏗️ BaseProducer 抽象基類
# ============================================================================

class BaseProducer(ABC):
    """Producer抽象基類"""
    
    def __init__(self, source, config: Optional[ProducerConfig] = None):
        """初始化Producer - 添加詳細logging"""
        
        # 初始化基本屬性
        self.source = source
        self.config = config or ProducerConfig()
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.cap = None
        
        # 添加Producer工廠初始化日誌
        logger.info(f"📝 來源輸入: {source}")
        logger.info(f"📝 Producer類型: {self.__class__.__name__}")
        logger.info(f"📝 模式識別: {self.mode}")
        logger.info(f"📝 配置參數: FPS間隔={self.config.fps_check_interval}, "
                f"緩衝大小={self.config.buffer_size}, 重試次數={self.config.retry_count}")
        
        # Template Method Pattern 初始化流程 logging
        logger.info("🔧 開始 Template Method 初始化流程...")
        logger.info("-" * 60)
        
        try:
            logger.info("📋 步驟 1/2: 初始化Capture物件...")
            self._initialize_capture()

            logger.info("📋 步驟 2/2: 配置系統串流參數...")
            self._configure_parameters()
            
            logger.info("🎉 Producer初始化完成!")
            
        except Exception as e:
            logger.error(f"❌ Producer初始化失敗: {e}")
            logger.error(f"❌ 失敗的Producer類型: {self.__class__.__name__}")
            logger.error(f"❌ 失敗的來源: {source}")
            raise CaptureInitializationError(f"Failed to initialize {self.__class__.__name__}: {e}")
        
        logger.info("🏭 " + "="*60)
    
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
    @property
    def mode(self) -> str:
        return "video"
    
    def _initialize_capture(self):
        """初始化Video文件capture"""
        # 文件存在性檢查
        import os
        if isinstance(self.source, str) and not os.path.exists(self.source):
            logger.error(f"❌ [VideoProducer] 文件不存在: {self.source}")
            raise RuntimeError(f"Video file not found: {self.source}")
        
        logger.info("🔍 [VideoProducer] 正在建立VideoCapture連接...")
        
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"❌ [VideoProducer] 無法開啟影片文件: {self.source}")
                raise RuntimeError(f"Cannot open video file: {self.source}")
            logger.info(f"📁 [VideoProducer] 文件載入完成: {self.source}")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.debug(f"🔧 [VideoProducer] 緩衝區大小設為: {self.config.buffer_size}")
            
        except Exception as e:
            logger.error(f"❌ [VideoProducer] Capture初始化失敗: {e}")
            logger.error(f"❌ [VideoProducer] 問題文件: {self.source}")
            raise
    
    def _configure_parameters(self):
        """配置Video特定參數"""
        # 基礎參數獲取
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"📊 [VideoProducer] FPS: {self.target_fps}")
        logger.info(f"📊 [VideoProducer] 解析度: {self.width}x{self.height}")
        logger.info(f"📊 [VideoProducer] 總幀數: {self.total_frames}")
        
        # Video模式特定屬性
        self.is_live_stream = False
        self.frame_duration = 1.0 / self.target_fps if self.target_fps > 0 else 0.033
        duration = self.total_frames / self.target_fps if self.target_fps > 0 else 0
        logger.info(f"⏱️ [VideoProducer] 影片時長: {duration:.2f} 秒")
        logger.info(f"⏱️ [VideoProducer] 每幀間隔: {self.frame_duration:.4f} 秒")
    
    def _get_next_frame(self):
        """Video幀獲取邏輯 - 添加詳細logging"""
        frame_start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            logger.info(f"🏁 [VideoProducer] 影片播放完成")
            logger.info(f"🏁 [VideoProducer] 總共處理幀數: {self.frame_counter}")
            logger.info(f"🏁 [VideoProducer] 完成率: 100% ({self.frame_counter}/{self.total_frames})")
            raise StopIteration
        
        self.frame_counter += 1
        
        # 每100幀記錄一次進度
        if self.frame_counter % 100 == 0:
            progress = (self.frame_counter / self.total_frames) * 100 if self.total_frames > 0 else 0
            logger.info(f"📈 [VideoProducer] 處理進度: {progress:.1f}% ({self.frame_counter}/{self.total_frames})")
        
        return frame

# ============================================================================
# 📷 CameraProducer 實現類
# ============================================================================

class CameraProducer(BaseProducer):
    @property
    def mode(self) -> str:
        return "camera"
    
    def _initialize_capture(self):
        """初始化Camera capture - 添加詳細logging"""
        camera_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        
        logger.info("📷 [CAMERA] =================================")
        logger.info("📷 [CAMERA] 開始初始化Camera實時Producer")
        logger.info("📷 [CAMERA] =================================")
        logger.info(f"📷 [CAMERA] 目標攝像頭 ID: {camera_id}")
        
        logger.info("🔍 [CAMERA] 正在搜尋並連接攝像頭...")
        
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logger.error(f"❌ [CAMERA] 無法連接攝像頭 ID: {camera_id}")
                logger.error(f"❌ [CAMERA] 可能原因: 攝像頭不存在、被其他程式使用、或驅動問題")
                raise RuntimeError(f"Cannot open camera: {camera_id}")
            
            logger.info("✅ [CAMERA] 攝像頭連接成功")
            logger.info(f"📹 [CAMERA] 攝像頭 ID {camera_id} 已就緒")
            
            # Camera模式優化設置
            logger.debug("🔧 [CAMERA] 應用Camera模式優化設置...")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.debug(f"🔧 [CAMERA] 緩衝區大小設為: {self.config.buffer_size} (減少延遲)")
            logger.info("✅ [CAMERA] Camera capture優化完成")
            
        except Exception as e:
            logger.error(f"❌ [CAMERA] Capture初始化失敗: {e}")
            logger.error(f"❌ [CAMERA] 問題攝像頭 ID: {camera_id}")
            raise CameraConnectionError(f"Failed to connect to camera {camera_id}: {e}")
    
    def _configure_parameters(self):
        """配置Camera特定參數 - 添加詳細logging"""
        logger.info("⚙️ [CAMERA] 開始配置Camera模式參數...")
        
        # 基礎參數獲取
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 [CAMERA] 攝像頭規格獲取完成:")
        logger.info(f"📊 [CAMERA]   - FPS: {self.target_fps}")
        logger.info(f"📊 [CAMERA]   - 解析度: {self.width}x{self.height}")
        
        # Camera模式特定屬性
        self.is_live_stream = True
        self.total_frames = -1  # 無限流
        self.connection_lost_count = 0
        self.init_time = time.time()  # 記錄初始化時間
        
        logger.info(f"🎯 [CAMERA] 模式設定: 實時性優先 (即時串流)")
        logger.info(f"⚡ [CAMERA] 串流特性: 無限幀數、自動重連、低延遲優化")
        logger.info(f"🔄 [CAMERA] 重連配置: 最大重試{self.config.retry_count}次, 間隔{self.config.reconnect_delay}秒")
        
        logger.info("✅ [CAMERA] Camera模式參數配置完成")
    
    def _get_next_frame(self):
        """Camera幀獲取邏輯 - 添加詳細logging"""
        frame_start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("⚠️ [CAMERA] 幀讀取失敗，可能是連接問題")
            logger.info("🔄 [CAMERA] 嘗試自動重連...")
            
            if self._attempt_reconnection():
                logger.info("✅ [CAMERA] 重連成功，繼續讀取幀")
                ret, frame = self.cap.read()
            
            if not ret:
                logger.error("❌ [CAMERA] 攝像頭連接永久丟失")
                logger.error(f"❌ [CAMERA] 總重連次數: {self.connection_lost_count}")
                raise CameraConnectionError("Camera connection lost permanently")
        
        self.frame_counter += 1
        return frame

    def _attempt_reconnection(self) -> bool:
        """嘗試重新連接Camera - 添加詳細logging"""
        logger.warning(f"🔄 [CAMERA] 檢測到連接中斷，開始重連程序...")
        logger.info(f"🔄 [CAMERA] 重連配置: 最大嘗試{self.config.retry_count}次, 每次間隔{self.config.reconnect_delay}秒")
        
        for attempt in range(self.config.retry_count):
            logger.info(f"🔄 [CAMERA] 重連嘗試 {attempt + 1}/{self.config.retry_count}")
            
            # 清理舊連接
            if self.cap:
                self.cap.release()
                logger.debug("🧹 [CAMERA] 舊連接已釋放")
            
            logger.debug(f"⏱️ [CAMERA] 等待 {self.config.reconnect_delay} 秒後重連...")
            time.sleep(self.config.reconnect_delay)
            
            # 嘗試重新連接
            try:
                camera_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
                logger.debug(f"🔍 [CAMERA] 嘗試重新連接攝像頭 ID {camera_id}...")
                
                self.cap = cv2.VideoCapture(camera_id)
                
                if self.cap.isOpened():
                    # 重新應用優化設置
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                    self.connection_lost_count += 1
                    
                    logger.info(f"✅ [CAMERA] 重連成功! (嘗試 {attempt + 1}/{self.config.retry_count})")
                    logger.info(f"📊 [CAMERA] 累計重連次數: {self.connection_lost_count}")
                    return True
                else:
                    logger.debug(f"❌ [CAMERA] 嘗試 {attempt + 1} 失敗: 無法開啟攝像頭")
                    
            except Exception as e:
                logger.warning(f"⚠️ [CAMERA] 重連嘗試 {attempt + 1} 發生異常: {e}")
        
        logger.error("❌ [CAMERA] 所有重連嘗試都失敗了")
        logger.error(f"❌ [CAMERA] 總共嘗試了 {self.config.retry_count} 次重連")
        logger.error(f"❌ [CAMERA] 攝像頭 ID {self.source} 可能已斷線或損壞")
        return False

# ============================================================================
# 🏭 工廠函數
# ============================================================================

def create_producer(source, config: Optional[ProducerConfig] = None) -> BaseProducer:
    """Producer工廠函數"""
    logger.info(f"🏭 [PRODUCER] 輸入來源: {source} (類型: {type(source).__name__})")
    # 自動判斷模式
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        producer = CameraProducer(source, config)
    else:
        producer = VideoProducer(source, config)
    return producer

class Producer(BaseProducer):
    """向後相容的Producer類"""
    def __new__(cls, source, filename=None, index=None, mode=None):
        logger.info("🔄 " + "="*60)
        logger.info("🔄 初始化Producer建構子")
        logger.info("🔄 " + "="*60)
        config = ProducerConfig()
        result = create_producer(source, config)
        return result
"""
================================================================================
🎬 Producer 架構設計 2025.08.21
================================================================================

Producer 採用抽象基類（BaseProducer）+ 工廠模式（Producer）的設計。
所有建立操作僅透過producer = Producer(source, mode) 作為唯一入口，並依據 mode 參數選擇 VideoProducer 或 CameraProducer。

系統支援兩種模式：
┌─────────────┬──────────────────┬─────────────────────────────────────────┐
│ 📸 Video    │ 完整性優先       │ 有限幀數、進度追蹤、完整性保證          │
│ 📷 Camera   │ 即時性優先       │ 無限流、自動重連、實時性優化            │
└─────────────┴──────────────────┴─────────────────────────────────────────┘

🆕 Frame ID 追蹤機制 (2025.08.23 新增)：
每個從 cap.read() 讀取的幀都會被分配一個連續的 frame_id，確保從 Producer 到 Consumer 的完整順序追蹤。
這個機制解決了多線程處理中可能出現的順序混亂問題，讓系統能夠追蹤每一幀從產生到顯示的完整生命周期。

📊 資料流向（更新版）：
    Video/Camera Source ──> Producer ──┬──> Frame + Frame_ID ──> Pipeline ──> WorkerPool ──> Consumer
                                        │
                                        └──> 順序追蹤保證

🎯 繼承關係：
                    BaseProducer (抽象基類)
                    ├── Template Method Pattern
                    ├── 統一初始化流程
                    ├── 通用監控與錯誤框架
                    ├── Frame ID 生成機制 🆕
                    └── 抽象方法定義
                           │
            ┌──────────────┴──────────────┐
            │                             │
    VideoProducer                  CameraProducer
    (完整性優先)                     (實時性優先)
    ├── 順序讀取Frame ID             ├── 實時Frame ID生成
    └── 進度追蹤 + ID對應             └── 重連保持ID連續性

📊 職責分配（◯ = 提供框架 / ✅ = 具體實作）：
┌─────────────────┬──────────────────┬─────────────────┬─────────────────┐
│   功能類別      │   BaseProducer   │  VideoProducer   │ CameraProducer  │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ 🏭 工廠創建     │ ◯ 抽象基類       │ 🔹 子類被動使用 │ 🔹 子類被動使用  │
│ 🚀 初始化管理   │ ✅ 模板方法流程   │ ◯ 文件驗證框架 │ ◯ 攝像頭偵測框架  │
│ 🎯 Capture設置  │ ✅ 通用設置       │ ✅ 讀取優化     │ ✅ 緩衝區配置   │
│ 📊 參數配置     │ ✅ 基礎參數框架   │ ✅ 進度追蹤     │ ✅ 實時參數      │
│ 🎬 幀生產邏輯   │ ◯ 抽象方法       │ ✅ 順序讀取     │ ✅ 實時捕獲      │ 
│ 🆕 Frame ID生成 │ ◯ ID計數器框架   │ ✅ 順序ID分配   │ ✅ 連續ID保證    │
│ 📈 性能監控     │ ◯ FPS 追蹤框架  │ ✅ 進度報告     │ ✅ 延遲監控       │ 
│ 🔄 錯誤處理     │ ◯ 基礎處理框架  │ ✅ 文件錯誤處理 │ ✅ 重連機制        │
│ 🧹 資源清理     │ ✅ 統一清理       │ 🔹 繼承使用     │ 🔹 繼承使用     │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┘

🛠️ 使用方式：
producer = Producer(args.video_path, mode=mode)

# 範例：
producer = Producer("./video.mp4", mode="VIDEO")      # 視頻文件處理 -> frame_id: 0,1,2,3...
producer = Producer("0", mode="CAMERA")               # 攝像頭即時流 -> frame_id: 0,1,2,3...
producer = Producer("rtsp://...", mode="CAMERA")      # RTSP 即時流 -> frame_id: 0,1,2,3...
"""

import cv2
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger('gstreamer_demo')

# ============================================================================
# 🔧 配置和異常導入
# ============================================================================
from utils.gstreamer.config import ProducerConfig
from utils.gstreamer.exceptions import CaptureInitializationError, FrameReadError, CameraConnectionError

# ============================================================================
# 🏗️ BaseProducer 抽象基類
# ============================================================================

class BaseProducer(ABC):
    """Producer抽象基類"""
    
    def __init__(self, source, config: Optional[ProducerConfig] = None):
        # 初始化基本屬性
        self.source = source
        self.config = config or ProducerConfig()
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.cap = None
        
        # Producer初始化日誌
        logger.info("🏭 " + "="*60)
        logger.info("🏭 Producer初始化開始")
        logger.info("🏭 " + "="*60)
        
        try:
            logger.info("📋 步驟 1/2: 🚀 初始化Capture物件...")
            self._initialize_capture()

            logger.info("📋 步驟 2/2: ⚙️ 配置管理 - 設置系統串流參數...")
            self._configure_parameters()

            logger.info("✅ Producer初始化完成!")

        except Exception as e:
            logger.error(f"❌ Producer初始化失敗: {e}")
            logger.error(f"❌ 失敗的Producer類型: {self.__class__.__name__}")
            logger.error(f"❌ 失敗的來源: {source}")
            if self.config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise CaptureInitializationError(f"Failed to initialize {self.__class__.__name__}: {e}")
    
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
        """開始幀迭代流程"""
        if self.config.detailed_logging:
            logger.info(f"🔄 [{self.mode.upper()}] 開始幀迭代流程")
        return self
    
    def __next__(self):
        """獲取下一幀 - 統一錯誤處理與記錄"""
        try:
            frame_data = self._get_next_frame()
            
            # 定期記錄性能統計
            if self.frame_counter % self.config.fps_check_interval == 0:
                self._log_performance_stats()
            
            return frame_data
            
        except StopIteration:
            self.cleanup()
            raise
            
        except Exception as e:
            logger.error(f"❌ [{self.mode.upper()}] 幀讀取失敗: {e}")
            if self.config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise FrameReadError(f"Failed to read frame: {e}")
    
    def _log_performance_stats(self):
        """記錄性能統計"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 0:
            current_fps = self.config.fps_check_interval / elapsed
            if self.config.detailed_logging:
                logger.info(f"📊 [{self.mode.upper()}] FPS={current_fps:.2f}, 幀數={self.frame_counter}")
        
        self.last_fps_time = current_time
    
    def cleanup(self):
        """資源清理"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info(f"🧹 [{self.mode.upper()}] 清理完成，總處理: {self.frame_counter}幀")
    
    def get_fps(self) -> float:
        """獲取目標FPS"""
        return getattr(self, 'target_fps', 30.0)
    
    def get_total_frames(self) -> int:
        """獲取總幀數"""
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
            logger.error(f"❌ [VIDEO] 文件不存在: {self.source}")
            raise RuntimeError(f"Video file not found: {self.source}")
        
        logger.info(f"🔍 [VIDEO] 目標文件: {self.source}")
        
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"❌ [VIDEO] 無法開啟影片文件: {self.source}")
                raise RuntimeError(f"Cannot open video file: {self.source}")

            # 應用Video模式優化設置
            logger.info("🔧 [VIDEO] 應用Video模式優化設置...")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.info(f"🔧 [VIDEO] 緩衝區大小設為: {self.config.buffer_size}")
            
        except Exception as e:
            logger.error(f"❌ [VIDEO] Capture初始化失敗: {e}")
            raise
    
    def _configure_parameters(self):
        """配置Video特定參數"""
        # 基礎參數獲取
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 [VIDEO] FPS: {self.target_fps}, 解析度: {self.width}x{self.height}, 總幀數: {self.total_frames}")
        
        # Video模式特定屬性
        self.is_live_stream = False
    
    def _get_next_frame(self):
        """Video幀獲取邏輯"""
        ret, frame = self.cap.read()
        if not ret:
            if self.frame_counter == 0:
                logger.error("❌ [VIDEO] 警告：沒有成功讀取任何幀！")
            raise StopIteration
        
        # 創建包含frame_id的數據結構
        frame_data = {
            'frame': frame,
            'frame_id': self.frame_counter,
            'timestamp': time.time(),
            'source': 'video',
            'mode': self.mode
        }
        
        self.frame_counter += 1
        return frame_data

# ============================================================================
# 📷 CameraProducer 實現類
# ============================================================================

class CameraProducer(BaseProducer):
    @property
    def mode(self) -> str:
        return "camera"
    
    def _initialize_capture(self):
        """初始化Camera capture"""
        camera_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        
        logger.info(f"📹 [CAMERA] 目標攝像頭 ID: {camera_id}")
        
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logger.error(f"❌ [CAMERA] 無法連接攝像頭 ID: {camera_id}")
                raise RuntimeError(f"Cannot open camera: {camera_id}")
            
            logger.info("✅ [CAMERA] 攝像頭連接成功")
            
            # Camera模式優化設置
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.info(f"🔧 [CAMERA] 緩衝區大小設為: {self.config.buffer_size}")
            
        except Exception as e:
            logger.error(f"❌ [CAMERA] Capture初始化失敗: {e}")
            raise CameraConnectionError(f"Failed to connect to camera {camera_id}: {e}")
    
    def _configure_parameters(self):
        """配置Camera特定參數"""
        # 基礎參數獲取
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 [CAMERA] FPS: {self.target_fps}, 解析度: {self.width}x{self.height}")
        
        # Camera模式特定屬性
        self.is_live_stream = True
        self.total_frames = -1
        self.connection_lost_count = 0
        self.init_time = time.time()
        
        logger.info(f"🎯 [CAMERA] 模式設定: 實時性優先, 重連配置: 最大{self.config.retry_count}次")
    
    def _get_next_frame(self):
        """Camera幀獲取邏輯"""
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("⚠️ [CAMERA] 讀取失敗，嘗試重連...")
            
            if self._attempt_reconnection():
                logger.info("✅ [CAMERA] 重連成功，恢復幀流")
                ret, frame = self.cap.read()
            
            if not ret:
                logger.error("❌ [CAMERA] 攝像頭連接永久丟失")
                raise CameraConnectionError("Camera connection lost permanently")
        
        # 創建包含frame_id的數據結構
        frame_data = {
            'frame': frame,
            'frame_id': self.frame_counter,
            'timestamp': time.time(),
            'source': 'camera',
            'mode': self.mode
        }
        
        self.frame_counter += 1
        return frame_data

    def _attempt_reconnection(self) -> bool:
        """嘗試重新連接Camera"""
        logger.warning(f"🔄 [CAMERA] 啟動重連程序，最大嘗試: {self.config.retry_count}次")
        
        for attempt in range(self.config.retry_count):
            logger.info(f"🔄 [CAMERA] 重連嘗試 {attempt + 1}/{self.config.retry_count}")
            
            # 清理舊連接
            if self.cap:
                self.cap.release()
            
            # 動態延遲
            delay = self.config.reconnect_delay * (attempt + 1)
            time.sleep(delay)
            
            # 嘗試重新連接
            try:
                camera_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
                self.cap = cv2.VideoCapture(camera_id)
                
                if self.cap.isOpened():
                    # 重新應用設置
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
                    self.connection_lost_count += 1
                    
                    logger.info(f"✅ [CAMERA] 重連成功! 累計重連次數: {self.connection_lost_count}")
                    return True
                    
            except Exception as e:
                logger.warning(f"⚠️ [CAMERA] 重連嘗試 {attempt + 1} 異常: {e}")
        
        logger.error(f"❌ [CAMERA] 所有重連嘗試失敗")
        return False

# ============================================================================
# 🏭 Producer 統一工廠類
# ============================================================================

class Producer:
    def __new__(cls, source, config: Optional[ProducerConfig] = None, mode: Optional[str] = None):
        """
        統一Producer創建接口
        
        Args:
            source: 輸入來源 (文件路徑或攝像頭ID)
            config: ProducerConfig配置對象 (可選)
            mode: 模式 ("VIDEO" 或 "CAMERA")
        """
        if mode is None:
            logger.error("❌ 缺少必要參數: mode")
            raise ValueError("必須指定 mode 參數 ('VIDEO' 或 'CAMERA')")
        
        if config is None:
            config = ProducerConfig()
        
        # 根據mode選擇Producer類型
        if mode == "VIDEO":
            return VideoProducer(source, config)
        elif mode == "CAMERA":
            return CameraProducer(source, config)
        else:
            logger.error(f"❌ 不支援的模式: {mode}")
            raise ValueError(f"不支援的模式: {mode}。請使用 'VIDEO' 或 'CAMERA'")
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

📊 資料流向：
    Video/Camera Source ──> Producer (自動採用對應的VideoProducer或CameraProducer)  ──> Pipeline

🎯 繼承關係：
                    BaseProducer (抽象基類)
                    ├── Template Method Pattern
                    ├── 統一初始化流程
                    ├── 通用監控與錯誤框架
                    └── 抽象方法定義
                           │
            ┌──────────────┴──────────────┐
            │                             │
    VideoProducer                  CameraProducer
    (完整性優先)                     (實時性優先)

📊 職責分配（◯ = 提供框架 / ✅ = 具體實作）：
┌─────────────────┬──────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   功能類別      │   BaseProducer   │  VideoProducer   │ CameraProducer   │   Producer      │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 🏭 工廠創建     │ ◯ 抽象基類       │ 🔹 子類被動使用 │ 🔹 子類被動使用 │ ✅ 統一入口     │
│ 🚀 初始化管理   │ ✅ 模板方法流程   │ ◯ 文件驗證框架 │ ◯ 攝像頭偵測框架 │ ✅ 類型選擇     │
│ 🎯 Capture設置  │ ✅ 通用設置       │ ✅ 讀取優化     │ ✅ 緩衝區配置   │ 🔹 委託實現     │
│ 📊 參數配置     │ ✅ 基礎參數框架   │ ✅ 進度追蹤     │ ✅ 實時參數     │ 🔹 委託實現     │
│ 🎬 幀生產邏輯   │ ◯ 抽象方法       │ ✅ 順序讀取     │ ✅ 實時捕獲     │ 🔹 委託實現     │
│ 📈 性能監控     │ ◯ FPS 追蹤框架  │ ✅ 進度報告     │ ✅ 延遲監控     │ 🔹 委託實現     │
│ 🔄 錯誤處理     │ ◯ 基礎處理框架  │ ✅ 文件錯誤處理 │ ✅ 重連機制       │ ✅ 統一處理     │
│ 🧹 資源清理     │ ✅ 統一清理       │ 🔹 繼承使用     │ 🔹 繼承使用     │ 🔹 委託實現     │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┴─────────────────┘

🛠️ 使用方式：
producer = Producer(args.video_path, mode=mode)

# 範例：
producer = Producer("./video.mp4", mode="video")      # 視頻文件處理
producer = Producer("0", mode="camera")               # 攝像頭即時流
producer = Producer("rtsp://...", mode="camera")      # RTSP 即時流
"""

import cv2
import logging
import time
import traceback
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
        """
        🏗️ Producer初始化 - Template Method Pattern
        
        Args:
            source: 輸入來源 (文件路徑或攝像頭ID)
            config: ProducerConfig配置對象
        """
        
        # 初始化基本屬性
        self.source = source
        self.config = config or ProducerConfig()
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.cap = None
        
        # 📝 Producer初始化日誌 - 簡潔統一格式
        logger.info("🏭 " + "="*60)
        logger.info("🏭 Producer初始化開始")
        logger.info("🏭 " + "="*60)
        
        try:
            logger.info("📋 步驟 1/2: 🚀 初始化Capture物件...")
            self._initialize_capture()

            logger.info("📋 步驟 2/2: ⚙️ 配置管理 - 設置系統串流參數...")
            self._configure_parameters()
            
            logger.info("✅ Producer初始化完成!")
            logger.info("🏭 " + "="*60)
            
        except Exception as e:
            logger.error(f"❌ Producer初始化失敗: {e}")
            logger.error(f"❌ 失敗的Producer類型: {self.__class__.__name__}")
            logger.error(f"❌ 失敗的來源: {source}")
            logger.error(f"❌ 錯誤分類: {type(e).__name__}")
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
        """
        🔄 開始幀迭代流程
        
        Returns:
            self: Iterator對象
        """
        if self.config.detailed_logging:
            logger.info(f"🔄 [{self.mode.upper()}] 開始幀迭代流程")
            logger.info(f"🔄 [{self.mode.upper()}] Producer類型: {self.__class__.__name__}")
        else:
            logger.debug(f"🔄 [{self.mode.upper()}] 開始幀迭代")
        return self
    
    def __next__(self):
        """
        📥 獲取下一幀 - 統一錯誤處理與記錄
        
        Returns:
            frame: 影像幀數據
            
        Raises:
            StopIteration: 當沒有更多幀時
            FrameReadError: 當讀取失敗時
        """
        try:
            frame = self._get_next_frame()
            
            # 每一定間隔記錄處理狀況
            if self.frame_counter % self.config.fps_check_interval == 0:
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                
                if elapsed > 0:
                    current_fps = self.config.fps_check_interval / elapsed
                    if self.config.detailed_logging:
                        logger.info(f"📊 [{self.mode.upper()}] 性能統計: "
                                   f"當前FPS={current_fps:.2f}, "
                                   f"已處理幀數={self.frame_counter}, "
                                   f"運行時長={elapsed:.1f}s")
                    else:
                        logger.debug(f"📊 [{self.mode.upper()}] FPS={current_fps:.2f}, 幀數={self.frame_counter}")
                
                self.last_fps_time = current_time
            
            return frame
            
        except StopIteration:
            if self.config.detailed_logging:
                logger.info(f"🏁 [{self.mode.upper()}] 幀迭代完成")
                logger.info(f"🏁 [{self.mode.upper()}] 總處理幀數: {self.frame_counter}")
                logger.info(f"🏁 [{self.mode.upper()}] Producer類型: {self.__class__.__name__}")
            else:
                logger.info(f"🏁 [{self.mode.upper()}] 完成，總幀數: {self.frame_counter}")
            self.cleanup()
            raise
            
        except Exception as e:
            logger.error(f"❌ [{self.mode.upper()}] 幀讀取失敗: {e}")
            logger.error(f"❌ [{self.mode.upper()}] 當前幀數: {self.frame_counter}")
            if self.config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise FrameReadError(f"Failed to read frame: {e}")
    
    def cleanup(self):
        """
        🧹 資源清理 - 線程安全的資源釋放
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            if self.config.detailed_logging:
                logger.info(f"🧹 [{self.mode.upper()}] Capture資源已釋放")
                logger.info(f"🧹 [{self.mode.upper()}] 清理完成，總處理: {self.frame_counter}幀")
            else:
                logger.debug(f"🧹 [{self.mode.upper()}] Capture資源已釋放")
    
    def get_fps(self) -> float:
        """
        📊 獲取目標FPS
        
        Returns:
            float: 目標幀率
        """
        return getattr(self, 'target_fps', 30.0)
    
    def get_total_frames(self) -> int:
        """
        📊 獲取總幀數
        
        Returns:
            int: 總幀數 (-1表示無限流)
        """
        return getattr(self, 'total_frames', -1)
    
    def get_stats(self) -> dict:
        """
        📊 獲取Producer統計資訊 - 統一接口
        
        Returns:
            dict: 統計資料字典
        """
        current_time = time.time()
        runtime = current_time - getattr(self, 'init_time', current_time)
        avg_fps = self.frame_counter / runtime if runtime > 0 else 0
        
        stats = {
            'producer_type': self.__class__.__name__,
            'mode': self.mode,
            'frame_counter': self.frame_counter,
            'target_fps': self.get_fps(),
            'actual_fps': avg_fps,
            'runtime_seconds': runtime,
            'is_live_stream': getattr(self, 'is_live_stream', False),
            'total_frames': self.get_total_frames(),
            'config': {
                'buffer_size': self.config.buffer_size,
                'timeout': self.config.timeout,
                'retry_count': self.config.retry_count,
                'reconnect_delay': self.config.reconnect_delay,
                'detailed_logging': self.config.detailed_logging
            }
        }
        
        # 模式特定統計
        if hasattr(self, 'connection_lost_count'):
            stats['connection_lost_count'] = self.connection_lost_count
            
        if self.get_total_frames() > 0:
            progress = (self.frame_counter / self.get_total_frames()) * 100
            stats['progress_percent'] = progress
        
        return stats

# ============================================================================
# 🎬 VideoProducer 實現類
# ============================================================================

class VideoProducer(BaseProducer):
    @property
    def mode(self) -> str:
        return "video"
    
    def _initialize_capture(self):
        """
        🚀 初始化Video文件capture - 完整性優先
        
        Video模式特點：
        • 文件完整性檢查
        • 讀取優化配置  
        • 進度追蹤準備
        """
        # 文件存在性檢查
        import os
        if isinstance(self.source, str) and not os.path.exists(self.source):
            logger.error(f"❌ [VIDEO] 文件不存在: {self.source}")
            logger.error(f"❌ [VIDEO] 請檢查文件路徑是否正確")
            raise RuntimeError(f"Video file not found: {self.source}")
        
        logger.info(f"🔍 [VIDEO] 目標文件: {self.source}")
        logger.info("🔍 [VIDEO] 正在建立VideoCapture連接...")
        
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"❌ [VIDEO] 無法開啟影片文件: {self.source}")
                logger.error(f"❌ [VIDEO] 可能原因: 文件格式不支援、文件損壞、或編解碼器問題")
                raise RuntimeError(f"Cannot open video file: {self.source}")

            # 應用Video模式優化設置
            logger.info("🔧 [VIDEO] 應用Video模式優化設置...")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.info(f"🔧 [VIDEO] 緩衝區大小設為: {self.config.buffer_size}")
            
        except Exception as e:
            logger.error(f"❌ [VIDEO] Capture初始化失敗: {e}")
            logger.error(f"❌ [VIDEO] 問題文件: {self.source}")
            if self.config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise
    
    def _configure_parameters(self):
        """
        ⚙️ 配置Video特定參數 - 完整性優先配置
        
        Video模式配置特點：
        • 進度追蹤支援
        • 完整性保證
        • 順序讀取優化
        """
        # 基礎參數獲取
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 [VIDEO]  - FPS: {self.target_fps}")
        logger.info(f"📊 [VIDEO]  - 解析度: {self.width}x{self.height}")
        logger.info(f"📊 [VIDEO]  - 總幀數: {self.total_frames}")
        
        # Video模式特定屬性
        self.is_live_stream = False
        self.frame_duration = 1.0 / self.target_fps if self.target_fps > 0 else 0.033
        duration = self.total_frames / self.target_fps if self.target_fps > 0 else 0
        
        logger.info(f"🎯 [VIDEO] 模式設定: 完整性優先")
        logger.info(f"⏱️ [VIDEO] 影片時長: {duration:.2f} 秒")
        logger.info(f"⏱️ [VIDEO] 每幀間隔: {self.frame_duration:.4f} 秒")
        logger.info(f"📈 [VIDEO] 進度追蹤: 啟用 (每100幀報告一次)")
    
    def _get_next_frame(self):
        """
        🎯 Video幀獲取邏輯 - 完整性優先處理
        
        Video模式特點：
        • 順序讀取保證
        • 進度追蹤報告  
        • 完整性檢查
        
        Returns:
            frame: 影像幀數據
            
        Raises:
            StopIteration: 當影片播放完成時
        """
        frame_start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            if self.config.detailed_logging:
                logger.info(f"🏁 [VIDEO] 影片播放完成")
                logger.info(f"🏁 [VIDEO] 總共處理幀數: {self.frame_counter}")
                logger.info(f"🏁 [VIDEO] 完成率: 100% ({self.frame_counter}/{self.total_frames})")
                logger.info(f"🏁 [VIDEO] Video模式任務完成")
            else:
                logger.info(f"🏁 [VIDEO] 播放完成，處理{self.frame_counter}幀")
            raise StopIteration
        
        self.frame_counter += 1
        
        # 進度追蹤報告 - Video模式特有
        if self.frame_counter % 100 == 0 or self.config.detailed_logging:
            progress = (self.frame_counter / self.total_frames) * 100 if self.total_frames > 0 else 0
            
            if self.config.detailed_logging:
                processing_time = time.time() - frame_start_time
                logger.info(f"📈 [VIDEO] 進度報告:")
                logger.info(f"📈 [VIDEO]   - 完成度: {progress:.1f}% ({self.frame_counter}/{self.total_frames})")
                logger.info(f"📈 [VIDEO]   - 當前幀處理時間: {processing_time*1000:.2f}ms")
            else:
                logger.info(f"📈 [VIDEO] 處理進度: {progress:.1f}% ({self.frame_counter}/{self.total_frames})")
        
        return frame

# ============================================================================
# 📷 CameraProducer 實現類
# ============================================================================

class CameraProducer(BaseProducer):
    @property
    def mode(self) -> str:
        return "camera"
    
    def _initialize_capture(self):
        """
        🚀 初始化Camera capture - 實時性優先  
        
        Camera模式特點：
        • 實時連接檢測
        • 低延遲優化
        • 自動重連準備
        """
        camera_id = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        
        logger.info(f"� [CAMERA] 目標攝像頭 ID: {camera_id}")
        logger.info("🔍 [CAMERA] 正在搜尋並連接攝像頭...")
        
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logger.error(f"❌ [CAMERA] 無法連接攝像頭 ID: {camera_id}")
                logger.error(f"❌ [CAMERA] 可能原因:")
                logger.error(f"❌ [CAMERA]   - 攝像頭不存在或未連接")
                logger.error(f"❌ [CAMERA]   - 被其他應用程式占用")
                logger.error(f"❌ [CAMERA]   - 驅動程式問題")
                logger.error(f"❌ [CAMERA]   - 權限不足")
                raise RuntimeError(f"Cannot open camera: {camera_id}")
            
            logger.info("✅ [CAMERA] 攝像頭連接成功")
            logger.info(f"📹 [CAMERA] 攝像頭 ID {camera_id} 已就緒")
            
            # Camera模式優化設置
            logger.debug("🔧 [CAMERA] 應用Camera模式優化設置...")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            logger.debug(f"🔧 [CAMERA] 緩衝區大小設為: {self.config.buffer_size} (實時性優化)")
            logger.info("✅ [CAMERA] Camera capture優化完成")
            
        except Exception as e:
            logger.error(f"❌ [CAMERA] Capture初始化失敗: {e}")
            logger.error(f"❌ [CAMERA] 問題攝像頭 ID: {camera_id}")
            if self.config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise CameraConnectionError(f"Failed to connect to camera {camera_id}: {e}")
    
    def _configure_parameters(self):
        """
        ⚙️ 配置Camera特定參數 - 實時性優先配置
        
        Camera模式配置特點：
        • 低延遲優化
        • 自動重連機制
        • 實時性監控
        """
        # 基礎參數獲取
        self.target_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 [CAMERA]  - FPS: {self.target_fps}")
        logger.info(f"📊 [CAMERA]  - 解析度: {self.width}x{self.height}")
        
        # Camera模式特定屬性
        self.is_live_stream = True
        self.total_frames = -1  # 無限流
        self.connection_lost_count = 0
        self.init_time = time.time()  # 記錄初始化時間
        
        logger.info(f"🎯 [CAMERA] 模式設定: 實時性優先")
        logger.info(f"⚡ [CAMERA] 串流特性: 無限幀數流、自動重連、低延遲")
        logger.info(f"🔄 [CAMERA] 重連配置: 最大{self.config.retry_count}次, 間隔{self.config.reconnect_delay}秒")
    
    def _get_next_frame(self):
        """
        🎯 Camera幀獲取邏輯 - 實時性優先處理
        
        Camera模式特點：
        • 實時性優先
        • 自動重連機制
        • 連接狀態監控
        
        Returns:
            frame: 影像幀數據
            
        Raises:
            CameraConnectionError: 當攝像頭連接永久丟失時
        """
        frame_start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            if self.config.detailed_logging:
                logger.warning("⚠️ [CAMERA] 幀讀取失敗，檢測到連接問題")
                logger.warning(f"⚠️ [CAMERA] 當前幀計數: {self.frame_counter}")
                logger.warning(f"⚠️ [CAMERA] 運行時間: {time.time() - self.init_time:.1f}秒")
            else:
                logger.warning("⚠️ [CAMERA] 讀取失敗，嘗試重連...")
            
            logger.info("🔄 [CAMERA] 啟動自動重連程序...")
            
            if self._attempt_reconnection():
                logger.info("✅ [CAMERA] 重連成功，恢復幀流")
                ret, frame = self.cap.read()
            
            if not ret:
                logger.error("❌ [CAMERA] 攝像頭連接永久丟失")
                logger.error(f"❌ [CAMERA] 總重連次數: {self.connection_lost_count}")
                logger.error(f"❌ [CAMERA] 運行時長: {time.time() - self.init_time:.1f}秒")
                logger.error(f"❌ [CAMERA] 處理幀數: {self.frame_counter}")
                raise CameraConnectionError("Camera connection lost permanently")
        
        self.frame_counter += 1
        
        # 實時性能監控
        if self.config.detailed_logging and self.frame_counter % self.config.fps_check_interval == 0:
            processing_time = time.time() - frame_start_time
            runtime = time.time() - self.init_time
            avg_fps = self.frame_counter / runtime if runtime > 0 else 0
            
            logger.debug(f"📊 [CAMERA] 實時性能監控:")
            logger.debug(f"📊 [CAMERA]   - 當前幀: {self.frame_counter}")
            logger.debug(f"📊 [CAMERA]   - 平均FPS: {avg_fps:.2f}")
            logger.debug(f"📊 [CAMERA]   - 幀處理時間: {processing_time*1000:.2f}ms")
            logger.debug(f"📊 [CAMERA]   - 運行時長: {runtime:.1f}s")
            logger.debug(f"📊 [CAMERA]   - 重連次數: {self.connection_lost_count}")
        
        return frame

    def _attempt_reconnection(self) -> bool:
        """
        🔄 嘗試重新連接Camera - 智能重連機制
        
        Camera模式重連特點：
        • 多次重試策略
        • 遞增延遲機制
        • 詳細狀態報告
        
        Returns:
            bool: 重連是否成功
        """
        logger.warning(f"🔄 [CAMERA] ======================================")
        logger.warning(f"🔄 [CAMERA] 檢測到連接中斷，啟動智能重連程序")
        logger.warning(f"🔄 [CAMERA] ======================================")
        logger.info(f"🔄 [CAMERA] 重連配置:")
        logger.info(f"🔄 [CAMERA]   - 最大嘗試: {self.config.retry_count}次")
        logger.info(f"🔄 [CAMERA]   - 基礎延遲: {self.config.reconnect_delay}秒")
        logger.info(f"🔄 [CAMERA]   - 歷史重連: {self.connection_lost_count}次")
        
        for attempt in range(self.config.retry_count):
            logger.info(f"🔄 [CAMERA] ------- 重連嘗試 {attempt + 1}/{self.config.retry_count} -------")
            
            # 清理舊連接
            if self.cap:
                self.cap.release()
                logger.debug("🧹 [CAMERA] 舊連接已釋放")
            
            # 計算動態延遲（遞增策略）
            delay = self.config.reconnect_delay * (attempt + 1)
            logger.debug(f"⏱️ [CAMERA] 等待 {delay} 秒後重連...")
            time.sleep(delay)
            
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
                    logger.info(f"✅ [CAMERA] 攝像頭 ID {camera_id} 已恢復連接")
                    logger.info(f"📊 [CAMERA] 累計重連次數: {self.connection_lost_count}")
                    logger.info(f"📊 [CAMERA] 總運行時間: {time.time() - self.init_time:.1f}秒")
                    
                    if self.config.detailed_logging:
                        logger.info(f"🔧 [CAMERA] 重連後配置:")
                        logger.info(f"🔧 [CAMERA]   - 緩衝區大小: {self.config.buffer_size}")
                        logger.info(f"🔧 [CAMERA]   - 處理幀數: {self.frame_counter}")
                    
                    return True
                else:
                    logger.debug(f"❌ [CAMERA] 嘗試 {attempt + 1} 失敗: 無法開啟攝像頭")
                    
            except Exception as e:
                logger.warning(f"⚠️ [CAMERA] 重連嘗試 {attempt + 1} 發生異常: {e}")
                if self.config.detailed_logging:
                    logger.warning(f"詳細錯誤: {traceback.format_exc()}")
        
        logger.error("❌ [CAMERA] =======================================")
        logger.error("❌ [CAMERA] 所有重連嘗試都失敗了")
        logger.error("❌ [CAMERA] =======================================")
        logger.error(f"❌ [CAMERA] 重連統計:")
        logger.error(f"❌ [CAMERA]   - 嘗試次數: {self.config.retry_count}")
        logger.error(f"❌ [CAMERA]   - 攝像頭ID: {self.source}")
        logger.error(f"❌ [CAMERA]   - 歷史重連: {self.connection_lost_count}")
        logger.error(f"❌ [CAMERA]   - 運行時長: {time.time() - self.init_time:.1f}秒")
        logger.error(f"❌ [CAMERA] 攝像頭可能已斷線、損壞或被其他程式占用")
        return False

# ============================================================================
# 🏭 Producer 統一工廠類 - 唯一創建入口
# ============================================================================

class Producer:
    def __new__(cls, source, config: Optional[ProducerConfig] = None, mode: Optional[str] = None):
        """
        🏭 統一Producer創建接口 - 唯一調用方式
        
        標準調用方式:
            producer = Producer(args.video_path, mode=mode)
        
        Args:
            source: 輸入來源 (文件路徑或攝像頭ID)
            config: ProducerConfig配置對象 (可選)
            mode: 模式 ("video" 或 "camera") - 必須指定
            
        Returns:
            BaseProducer: VideoProducer或CameraProducer實例
            
        Raises:
            ValueError: mode參數不正確或未指定時
        """
        # 檢查必要參數
        if mode is None:
            logger.error("❌ 缺少必要參數: mode")
            logger.error("❌ 正確調用方式: Producer(source, mode='video') 或 Producer(source, mode='camera')")
            raise ValueError("必須指定 mode 參數 ('video' 或 'camera')")
        
        # 處理配置對象
        if config is None:
            config = ProducerConfig()
        
        # 🎯 根據mode參數選擇Producer類型
        if mode == "video":
            return VideoProducer(source, config)
        elif mode == "camera":
            return CameraProducer(source, config)
        else:
            logger.error(f"❌ 不支援的模式: {mode}")
            logger.error("❌ 支援的模式: 'video' 或 'camera'")
            raise ValueError(f"不支援的模式: {mode}。請使用 'video' 或 'camera'")
from dataclasses import dataclass
from typing import Literal

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
    
    # 記錄詳細程度
    detailed_logging: bool = False

@dataclass
class ProcessorConfig:
    """Processor推論行為控制配置類 - 支援mode參數控制Video/Camera模式差異"""
    mode: Literal['VIDEO', 'CAMERA'] = 'CAMERA'
    
    # VIDEO模式：完整性優先 - 寬鬆超時、詳細記錄
    # CAMERA模式：實時性優先 - 嚴格超時、精簡記錄
    
    # 推論超時控制
    inference_timeout: float = 5.0  # CAMERA模式嚴格超時，VIDEO模式放寬
    warmup_timeout: float = 10.0
    
    # 記錄詳細程度
    detailed_logging: bool = False  # CAMERA精簡，VIDEO詳細
    
    # 模型路徑
    model_path: str = ""
    
    def __post_init__(self):
        """根據mode自動調整參數"""
        if self.mode == 'VIDEO':
            # VIDEO模式：完整性優先 - 寬鬆超時、詳細記錄
            if self.inference_timeout == 5.0:  # 如果是默認值
                self.inference_timeout = 15.0
            self.detailed_logging = True
        elif self.mode == 'CAMERA':
            # CAMERA模式：實時性優先 - 嚴格超時、精簡記錄  
            if self.inference_timeout == 15.0:  # 如果是VIDEO設定
                self.inference_timeout = 5.0
            self.detailed_logging = False

@dataclass
class WorkerPoolConfig:
    """WorkerPool調度策略控制配置類 - 支援mode參數控制VIDEO/CAMERA模式差異"""
    mode: Literal['VIDEO', 'CAMERA'] = 'CAMERA'
    
    # VIDEO模式：順序保證、大緩衝、無丟幀、停用背壓
    # CAMERA模式：背壓控制、小緩衝、即時策略、啟用背壓
    
    # 工作池基本配置
    max_workers: int = 4
    
    # 緩衝區控制
    buffer_size: int = 5  # CAMERA小緩衝，VIDEO大緩衝
    
    # 背壓控制
    enable_backpressure: bool = True  # CAMERA啟用，VIDEO停用
    drop_threshold: float = 0.8  # 背壓觸發閾值
    
    # 順序保證（VIDEO專用）
    preserve_order: bool = False  # CAMERA無需順序，VIDEO需要順序
    
    # 調度策略
    scheduling_strategy: str = 'round_robin'
    
    def __post_init__(self):
        """根據mode自動調整參數"""
        if self.mode == 'VIDEO':
            # VIDEO模式：順序保證、大緩衝、無丟幀、停用背壓
            self.buffer_size = max(self.buffer_size, 20)  # 大緩衝
            self.enable_backpressure = False  # 停用背壓
            self.preserve_order = True  # 順序保證
        elif self.mode == 'CAMERA':
            # CAMERA模式：背壓控制、小緩衝、即時策略、啟用背壓
            self.buffer_size = min(self.buffer_size, 5)  # 小緩衝
            self.enable_backpressure = True  # 啟用背壓
            self.preserve_order = False  # 無需順序
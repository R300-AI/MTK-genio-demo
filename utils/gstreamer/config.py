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
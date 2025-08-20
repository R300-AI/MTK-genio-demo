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
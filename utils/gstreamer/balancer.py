import threading
import time
import logging
from collections import deque

logger = logging.getLogger('gstreamer_demo')

class Balancer:
    """
    智慧型 Pipeline 速率平衡器，根據實際處理能力動態調整 Producer 速率。
    監控 Produced/Processed/Consumed FPS，自動調節 Producer 的幀間延遲。
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.monitor = None  # 將由外部設定
        self.producer = None  # 將由外部設定
        logger.info(f"[BALANCER] Initialized.")

    def set_monitor(self, monitor):
        """設定 Monitor 實例"""
        self.monitor = monitor

    def set_producer(self, producer):
        """設定 Producer 實例"""
        self.producer = producer

    def get_producer_sleep(self, current_frame_interval):
        """根據 processed_fps 決定最佳 frame_interval，無需任何狀態。"""
        _, _, processed_fps = self._get_metrics()
        try:
            processed_fps_num = float(processed_fps)
            if processed_fps_num > 0:
                interval = 1.0 / processed_fps_num
                if abs(interval - current_frame_interval) > 0.001:
                    logger.info(f"[BALANCER] Producer interval auto-adjust: {current_frame_interval:.3f}s -> {interval:.3f}s (target by processed_fps={processed_fps_num:.2f})")
                return interval
        except (TypeError, ValueError):
            pass
        return current_frame_interval

    def _get_metrics(self):
        """從 Monitor 取得當前指標，直接使用 Monitor 的 FPS 計算"""
        if not self.monitor:
            return None, None, None
        with self.monitor.lock:
            queued = self.monitor.frame_count - self.monitor.processed_count
            produced_fps = self.monitor.produced_fps
            processed_fps = self.monitor.processed_fps
            return queued, produced_fps, processed_fps
#!/usr/bin/env python3
"""
測試Pipeline的詳細logging功能
"""

import time
from utils.gstreamer.pipeline import create_pipeline


class MockProducer:
    """模擬Producer"""
    def __init__(self, mode='video'):
        self.mode = mode
        self.frames = [f"frame_{i}" for i in range(3)]  # 少量幀用於測試
    
    def __iter__(self):
        for frame in self.frames:
            time.sleep(0.1)  # 模擬處理時間
            yield frame


class MockWorkerPool:
    """模擬WorkerPool"""
    def __init__(self):
        self.active_workers = 0
        self.num_workers = 2
    
    def start(self, callback):
        self.callback = callback
    
    def submit(self, frame):
        # 模擬處理並返回結果
        result = f"processed_{frame}"
        if hasattr(self, 'callback'):
            self.callback(result)
    
    def stop(self):
        pass


class MockConsumer:
    """模擬Consumer"""
    def __init__(self):
        self.consumed_count = 0
    
    def start_display(self):
        pass
    
    def consume(self, result):
        self.consumed_count += 1
    
    def stop_display(self):
        pass


def test_pipeline_logging():
    """測試Pipeline詳細logging"""
    print("📝 測試Pipeline詳細logging...")
    print("=" * 60)
    
    # 測試Video模式logging
    print("\n📹 測試Video模式Pipeline logging:")
    video_producer = MockProducer(mode='video')
    video_worker = MockWorkerPool()
    video_consumer = MockConsumer()
    
    video_pipeline = create_pipeline(video_producer, video_worker, video_consumer)
    print("✅ Video Pipeline 創建完成，請檢查 gstreamer_demo.log 文件")
    
    # 測試Camera模式logging
    print("\n📸 測試Camera模式Pipeline logging:")
    camera_producer = MockProducer(mode='camera')
    camera_worker = MockWorkerPool()
    camera_consumer = MockConsumer()
    
    camera_pipeline = create_pipeline(camera_producer, camera_worker, camera_consumer)
    print("✅ Camera Pipeline 創建完成，請檢查 gstreamer_demo.log 文件")
    
    print(f"\n🎉 測試完成！請查看 'gstreamer_demo.log' 檢視詳細logging內容")


if __name__ == "__main__":
    test_pipeline_logging()

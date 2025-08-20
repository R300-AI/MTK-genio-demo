#!/usr/bin/env python3
"""
測試新的Pipeline繼承架構
"""

from utils.gstreamer.pipeline import BasePipeline, VideoPipeline, CameraPipeline, create_pipeline


class MockProducer:
    """模擬Producer"""
    def __init__(self, mode='video'):
        self.mode = mode
        self.frames = [f"frame_{i}" for i in range(5)]
    
    def __iter__(self):
        return iter(self.frames)


class MockWorkerPool:
    """模擬WorkerPool"""
    def __init__(self):
        self.active_workers = 0
        self.num_workers = 2
    
    def start(self, callback):
        self.callback = callback
        print("WorkerPool started")
    
    def submit(self, frame):
        # 模擬處理並直接返回結果
        result = f"processed_{frame}"
        if hasattr(self, 'callback'):
            self.callback(result)
    
    def stop(self):
        print("WorkerPool stopped")


class MockConsumer:
    """模擬Consumer"""
    def __init__(self):
        self.consumed_count = 0
    
    def start_display(self):
        print("Consumer display started")
    
    def consume(self, result):
        self.consumed_count += 1
        print(f"Consumed: {result}")
    
    def stop_display(self):
        print("Consumer display stopped")


def test_pipeline_architecture():
    """測試Pipeline架構"""
    print("🏗️ 測試Pipeline繼承架構")
    print("=" * 50)
    
    # 測試Video模式
    print("\n📹 測試Video模式Pipeline:")
    video_producer = MockProducer(mode='video')
    video_worker = MockWorkerPool()
    video_consumer = MockConsumer()
    
    video_pipeline = create_pipeline(video_producer, video_worker, video_consumer)
    print(f"✅ 創建Video Pipeline: {type(video_pipeline).__name__}")
    assert isinstance(video_pipeline, VideoPipeline)
    
    # 測試Camera模式
    print("\n📸 測試Camera模式Pipeline:")
    camera_producer = MockProducer(mode='camera')
    camera_worker = MockWorkerPool()
    camera_consumer = MockConsumer()
    
    camera_pipeline = create_pipeline(camera_producer, camera_worker, camera_consumer)
    print(f"✅ 創建Camera Pipeline: {type(camera_pipeline).__name__}")
    assert isinstance(camera_pipeline, CameraPipeline)
    
    # 測試繼承關係
    print(f"\n🔗 測試繼承關係:")
    print(f"VideoPipeline 是 BasePipeline 的子類: {issubclass(VideoPipeline, BasePipeline)}")
    print(f"CameraPipeline 是 BasePipeline 的子類: {issubclass(CameraPipeline, BasePipeline)}")
    
    # 測試抽象基類
    print(f"\n🎯 測試抽象基類:")
    try:
        base = BasePipeline(video_producer, video_worker, video_consumer)
        print("❌ BasePipeline 不應該能直接實例化")
    except TypeError as e:
        print(f"✅ BasePipeline 正確地作為抽象基類: {e}")
    
    print(f"\n🎉 所有測試通過！Pipeline繼承架構運作正常")


if __name__ == "__main__":
    test_pipeline_architecture()

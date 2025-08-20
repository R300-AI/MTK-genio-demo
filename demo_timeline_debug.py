#!/usr/bin/env python3
"""
Timeline Debug 功能示範腳本
展示詳細的時間軸調試功能，用於分析worker分配問題
"""

import sys
import os
import time
import threading
sys.path.append(os.path.dirname(__file__))

from utils.gstreamer.pipeline import Pipeline, TimelineDebugger
import random

class MockProducer:
    """模擬Producer，用於測試timeline debugging"""
    def __init__(self, mode="video", total_frames=50):
        self.mode = mode
        self.total_frames = total_frames
        self.current_frame = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_frame >= self.total_frames:
            raise StopIteration
        
        self.current_frame += 1
        # 模擬不同的產生速率
        time.sleep(random.uniform(0.02, 0.08))  # 20-80ms變動
        return f"frame_{self.current_frame}"

class MockWorkerPool:
    """模擬WorkerPool，用於測試timeline debugging"""
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.active_workers = 0
        self.processing_tasks = 0
        self.result_handler = None
        self.workers = []
        
        # 創建模擬worker對象
        for i in range(num_workers):
            worker = type('Worker', (), {
                'is_busy': False,
                'task_count': 0,
                'id': i+1
            })()
            self.workers.append(worker)
    
    def start(self, result_handler):
        self.result_handler = result_handler
        print(f"[MockWorkerPool] Started with {self.num_workers} workers")
    
    def process(self, frame):
        """模擬處理frame，包含可能的worker分配問題"""
        self.processing_tasks += 1
        
        # 模擬worker分配邏輯
        available_workers = [w for w in self.workers if not w.is_busy]
        
        if available_workers:
            worker = available_workers[0]
            worker.is_busy = True
            worker.task_count += 1
            self.active_workers += 1
            
            def process_frame():
                try:
                    # 模擬處理時間變動，包含可能的"批次同步"問題
                    if random.random() < 0.1:  # 10%機率出現處理延遲
                        time.sleep(random.uniform(0.1, 0.3))  # 100-300ms延遲
                    else:
                        time.sleep(random.uniform(0.03, 0.07))  # 正常30-70ms
                    
                    # 處理完成
                    result = f"processed_{frame}"
                    if self.result_handler:
                        self.result_handler(result)
                        
                finally:
                    worker.is_busy = False
                    self.active_workers -= 1
                    self.processing_tasks -= 1
            
            # 啟動處理線程
            thread = threading.Thread(target=process_frame, daemon=True)
            thread.start()
            
        else:
            # 模擬worker都忙碌的情況
            print(f"[MockWorkerPool] All workers busy, dropping frame: {frame}")
    
    def stop(self):
        print("[MockWorkerPool] Stopped")

class MockConsumer:
    """模擬Consumer，用於測試timeline debugging"""
    def __init__(self):
        self.processed_results = 0
        self.display_active = False
    
    def start_display(self):
        self.display_active = True
        print("[MockConsumer] Display started")
    
    def put_frame(self, result):
        # 模擬顯示處理時間
        time.sleep(random.uniform(0.02, 0.05))  # 20-50ms
        self.processed_results += 1
        if self.processed_results % 10 == 0:
            print(f"[MockConsumer] Processed {self.processed_results} results")
    
    def stop_display(self):
        self.display_active = False
        print(f"[MockConsumer] Display stopped, total processed: {self.processed_results}")

class MockMonitor:
    """模擬Monitor，用於追蹤處理狀態"""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_count = 0
        self.processed_count = 0
        self.consumed_count = 0
        self.processing = 0

def demonstrate_timeline_debugging():
    """示範timeline debugging功能"""
    print("\n🕐 Timeline Debug 功能示範")
    print("="*60)
    
    # 創建模擬組件
    producer = MockProducer(mode="video", total_frames=30)  # 較少幀數便於觀察
    worker_pool = MockWorkerPool(num_workers=2)
    consumer = MockConsumer()
    monitor = MockMonitor()
    
    print("創建Pipeline實例...")
    pipeline = Pipeline(producer, worker_pool, consumer, monitor)
    
    print("\n啟用密集調試模式...")
    pipeline.enable_intensive_debug_logging()
    
    print("\n開始運行Pipeline (將產生詳細的時間軸調試信息)...")
    print("請查看log文件中的 [TIMELINE-DEBUG] 和相關調試信息")
    print("-" * 60)
    
    # 在另一個線程中定期打印摘要
    def periodic_summary():
        time.sleep(2)  # 等待Pipeline開始
        for i in range(6):  # 運行期間打印6次摘要
            time.sleep(3)
            if pipeline.running:
                print(f"\n📊 運行中摘要 (第{i+1}次):")
                pipeline.log_current_debug_state()
                print(pipeline.get_debug_timeline_summary(last_n_seconds=5))
    
    summary_thread = threading.Thread(target=periodic_summary, daemon=True)
    summary_thread.start()
    
    try:
        # 運行Pipeline
        start_time = time.time()
        pipeline.run()
        end_time = time.time()
        
        print(f"\n✅ Pipeline運行完成，總耗時: {end_time - start_time:.2f}秒")
        
        # 顯示最終的詳細時間軸
        print("\n🎯 最終時間軸分析:")
        print("-" * 60)
        pipeline.print_debug_timeline(last_n_snapshots=25)
        
        print("\n📈 最終摘要:")
        print(pipeline.get_debug_timeline_summary(last_n_seconds=30))
        
    except Exception as e:
        print(f"❌ 運行過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_standalone_timeline_debugger():
    """示範獨立的TimelineDebugger功能"""
    print("\n🔧 獨立Timeline Debugger示範")
    print("="*50)
    
    debugger = TimelineDebugger(max_history=50)
    
    print("模擬Pipeline運行狀態變化...")
    
    # 模擬10秒的Pipeline運行
    for t in range(100):  # 100個時間點，每個100ms
        # 模擬Producer狀態
        producer_active = t < 80  # 前8秒Producer活動
        debugger.update_producer_state(active=producer_active, frame_count=t if producer_active else 80)
        
        # 模擬Queue狀態變化
        input_queue = max(0, min(50, t//2 + random.randint(-5, 5)))  # 隊列大小變化
        output_queue = max(0, min(50, (t-10)//3 + random.randint(-3, 3)))  # 輸出隊列延遲
        debugger.update_queue_states(input_size=input_queue, output_size=output_queue, 
                                    input_max=50, output_max=50)
        
        # 模擬Worker狀態（包含週期性的worker分配為0問題）
        for worker_id in ['W1', 'W2']:
            # 模擬週期性worker問題（每20個時間點會有5個時間點worker為0）
            cycle_pos = t % 20
            if 12 <= cycle_pos <= 16:  # 週期性的worker餓死問題
                worker_active = False
            else:
                worker_active = random.random() > 0.3  # 70%機率活動
            
            debugger.update_worker_state(worker_id, active=worker_active, task_count=t//5)
        
        # 模擬Consumer狀態
        consumer_active = t > 5 and t < 90  # Consumer稍晚開始，稍早結束
        debugger.update_consumer_state(active=consumer_active, result_count=max(0, t-5))
        
        # 記錄時間軸
        debugger.log_timeline_snapshot()
        
        time.sleep(0.01)  # 10ms間隔
    
    print("\n📊 模擬運行完成，顯示時間軸分析:")
    print(debugger.get_timeline_summary(last_n_seconds=10))
    
    print("\n🎯 視覺化時間軸:")
    debugger.print_visual_timeline(last_n_snapshots=30)

def main():
    """主示範函數"""
    print("🔍 MTK Genio Pipeline Timeline Debug 功能示範")
    print("="*70)
    
    choice = input("\n選擇示範模式:\n1. 完整Pipeline + Timeline Debug\n2. 獨立Timeline Debugger\n請輸入 (1 或 2): ").strip()
    
    if choice == "2":
        demonstrate_standalone_timeline_debugger()
    else:
        demonstrate_timeline_debugging()
    
    print("\n" + "="*70)
    print("📝 Timeline Debug功能說明:")
    print("• [TIMELINE-DEBUG] - 每500ms的時間軸快照")
    print("• [PRODUCER-DEBUG] - Producer詳細操作記錄") 
    print("• [WORKER-DEBUG] - Worker處理詳細記錄")
    print("• [CONSUMER-DEBUG] - Consumer處理詳細記錄")
    print("• [TIMELINE-ALERT] - 檢測到的問題警告")
    print("• 視覺化時間軸顯示各組件的活動狀態")
    print("• 可以精確分析worker分配為0的時間點和原因")
    print("="*70)
    
    print("\n✅ 示範完成！查看 gstreamer_demo.log 獲取詳細的時間軸調試信息")

if __name__ == "__main__":
    main()

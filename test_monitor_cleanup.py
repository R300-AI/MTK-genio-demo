#!/usr/bin/env python3
"""測試Monitor類的方法調用是否正確"""

import sys
import os

# 添加當前目錄到 path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.gstreamer.monitor import Monitor

def test_monitor_methods():
    """測試Monitor方法是否存在並可以調用"""
    print("開始測試 Monitor 類的方法...")
    
    # 創建 Monitor 實例
    monitor = Monitor(window_name="test", max_fps=30)
    
    # 測試核心方法是否存在
    core_methods = [
        'count_produced',
        'count_processing_start', 
        'count_processing_end',
        'count_processed',
        'count_consumed',
        '_check_and_log',
        '_log_full_status'
    ]
    
    print("檢查核心方法是否存在:")
    for method in core_methods:
        if hasattr(monitor, method):
            print(f"✓ {method} - 存在")
        else:
            print(f"✗ {method} - 不存在")
    
    # 測試已刪除的方法是否確實被移除
    removed_methods = [
        'increment_produced',
        'increment_processing',
        'decrement_processing', 
        'increment_processed',
        'increment_consumed'
    ]
    
    print("\n檢查已刪除的包裝方法:")
    for method in removed_methods:
        if hasattr(monitor, method):
            print(f"✗ {method} - 仍然存在 (應該被刪除)")
        else:
            print(f"✓ {method} - 已成功刪除")
    
    # 測試方法調用
    print("\n測試方法調用:")
    try:
        monitor.count_produced()
        print("✓ count_produced() - 調用成功")
    except Exception as e:
        print(f"✗ count_produced() - 調用失敗: {e}")
    
    try:
        monitor.count_processing_start()
        print("✓ count_processing_start() - 調用成功")
    except Exception as e:
        print(f"✗ count_processing_start() - 調用失敗: {e}")
        
    try:
        monitor.count_processing_end()
        print("✓ count_processing_end() - 調用成功")
    except Exception as e:
        print(f"✗ count_processing_end() - 調用失敗: {e}")
        
    try:
        monitor.count_processed()
        print("✓ count_processed() - 調用成功")
    except Exception as e:
        print(f"✗ count_processed() - 調用失敗: {e}")
        
    try:
        monitor.count_consumed()
        print("✓ count_consumed() - 調用成功")
    except Exception as e:
        print(f"✗ count_consumed() - 調用失敗: {e}")
    
    # 測試狀態
    print(f"\n當前狀態:")
    print(f"  produced: {monitor.produced_count}")
    print(f"  processing: {monitor.processing_count}")
    print(f"  processed: {monitor.processed_count}")
    print(f"  consumed: {monitor.consumed_count}")
    
    print("\n✓ Monitor 類測試完成！")

if __name__ == "__main__":
    test_monitor_methods()

#!/usr/bin/env python3
"""
背壓控制效果測試腳本
比較啟用背壓控制前後的處理平滑度
"""

import subprocess
import time
import os
import re

def run_test(test_name, args, duration=30):
    """執行測試並分析結果"""
    print(f"\n{'='*60}")
    print(f"🧪 測試: {test_name}")
    print(f"{'='*60}")
    
    cmd = ["python", "gstreamer_demo.py"] + args
    print(f"執行指令: {' '.join(cmd)}")
    
    log_file = "gstreamer_demo.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print(f"運行 {duration} 秒...")
        
        # 讓程式運行指定時間
        time.sleep(duration)
        
        # 終止程式
        process.terminate()
        process.wait(timeout=5)
        
    except Exception as e:
        print(f"❌ 執行出錯: {e}")
        return None
    
    # 分析結果
    return analyze_performance(test_name)

def analyze_performance(test_name):
    """分析性能日誌"""
    log_file = "gstreamer_demo.log"
    
    if not os.path.exists(log_file):
        print("📄 找不到日誌檔案")
        return None
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️ 讀取日誌失敗: {e}")
        return None
    
    # 統計指標
    stats = {
        'total_frames_produced': 0,
        'total_frames_processed': 0,
        'total_frames_consumed': 0,
        'balancer_adjustments': 0,
        'dropped_frames': 0,
        'backpressure_activations': 0
    }
    
    for line in lines:
        # 統計產生的幀數
        if "Produced:" in line:
            match = re.search(r'Produced: (\d+)', line)
            if match:
                stats['total_frames_produced'] = max(stats['total_frames_produced'], int(match.group(1)))
        
        # 統計處理的幀數  
        if "Processed:" in line:
            match = re.search(r'Processed: (\d+)', line)
            if match:
                stats['total_frames_processed'] = max(stats['total_frames_processed'], int(match.group(1)))
        
        # 統計消費的幀數
        if "Consumed:" in line:
            match = re.search(r'Consumed: (\d+)', line)
            if match:
                stats['total_frames_consumed'] = max(stats['total_frames_consumed'], int(match.group(1)))
        
        # 統計平衡器調整次數
        if "Producer interval auto-adjust" in line:
            stats['balancer_adjustments'] += 1
        
        # 統計丟幀次數
        if "dropping frame" in line:
            stats['dropped_frames'] += 1
        
        # 統計背壓啟動次數
        if "System overloaded" in line:
            stats['backpressure_activations'] += 1
    
    # 計算處理效率
    if stats['total_frames_produced'] > 0:
        processing_efficiency = (stats['total_frames_processed'] / stats['total_frames_produced']) * 100
    else:
        processing_efficiency = 0
    
    print(f"\n📊 {test_name} 性能分析:")
    print(f"  📥 總產生幀數: {stats['total_frames_produced']}")
    print(f"  ⚙️  總處理幀數: {stats['total_frames_processed']}")
    print(f"  📺 總消費幀數: {stats['total_frames_consumed']}")
    print(f"  🔄 平衡器調整: {stats['balancer_adjustments']} 次")
    print(f"  🗑️ 丟棄幀數: {stats['dropped_frames']}")
    print(f"  🚫 背壓啟動: {stats['backpressure_activations']} 次")
    print(f"  📈 處理效率: {processing_efficiency:.1f}%")
    
    return stats

def compare_results(baseline, improved):
    """比較兩個測試結果"""
    if not baseline or not improved:
        print("⚠️ 無法比較結果")
        return
    
    print(f"\n📊 對比分析:")
    
    # 計算改善
    balance_improvement = baseline['balancer_adjustments'] - improved['balancer_adjustments']
    if baseline['balancer_adjustments'] > 0:
        balance_percentage = (balance_improvement / baseline['balancer_adjustments']) * 100
    else:
        balance_percentage = 0
    
    print(f"  🔄 平衡器調整減少: {balance_improvement} 次 ({balance_percentage:+.1f}%)")
    print(f"  🗑️ 丟幀情況: {improved['dropped_frames']} 次")
    print(f"  🚫 背壓控制啟動: {improved['backpressure_activations']} 次")
    
    if improved['backpressure_activations'] > 0:
        print("  ✅ 背壓控制正常工作")
    else:
        print("  ⚠️ 背壓控制未啟動（可能負載不夠）")

def main():
    print("🚀 背壓控制效果測試")
    print("本腳本將比較啟用背壓控制前後的效果")
    
    test_duration = 20  # 每個測試運行20秒
    
    tests = [
        {
            "name": "基準測試 (停用平衡器)",
            "args": ["--video_path", "./data/video.mp4", "--max_workers", "4", "--disable_balancer"]
        },
        {
            "name": "背壓控制 (啟用平衡器)",
            "args": ["--video_path", "./data/video.mp4", "--max_workers", "4", "--enable_backpressure_logging"]
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n🔄 準備執行: {test['name']}")
        input("按 Enter 繼續...")
        
        result = run_test(test["name"], test["args"], test_duration)
        results.append(result)
    
    # 比較結果
    if len(results) >= 2:
        compare_results(results[0], results[1])
    
    print(f"\n✅ 測試完成!")
    print(f"\n📝 觀察重點:")
    print(f"  1. 背壓控制是否有效減少平衡器調整頻率")
    print(f"  2. 丟幀機制是否在高負載時正常工作")
    print(f"  3. 整體處理流程是否更加平滑")

if __name__ == "__main__":
    main()

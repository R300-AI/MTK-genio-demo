#!/usr/bin/env python3
"""
硬體適應性 Pipeline 示範腳本
展示不同硬體性能等級的自動檢測和參數調整功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.gstreamer.pipeline import Pipeline, HardwarePerformanceDetector
import time

def demo_hardware_detection():
    """示範硬體性能檢測功能"""
    print("\n🔍 硬體性能檢測示範")
    print("="*50)
    
    # 創建硬體檢測器
    detector = HardwarePerformanceDetector()
    
    print(f"檢測到的硬體性能等級: {detector.performance_tier}")
    print(f"CPU核心數: {detector.cpu_cores} ({detector.cpu_physical_cores} 物理核心)")
    print(f"記憶體容量: {detector.memory_gb:.2f} GB")
    
    if detector.benchmark_results:
        print(f"CPU基準測試分數: {detector.benchmark_results.get('cpu_score', 'N/A'):.2f}")
        print(f"記憶體基準測試分數: {detector.benchmark_results.get('memory_score', 'N/A'):.2f}")
    
    # 顯示適應性參數
    params = detector.get_adaptive_parameters()
    print(f"\n⚙️ 針對此硬體的適應性參數:")
    for key, value in params.items():
        if isinstance(value, float):
            if key.endswith('timeout'):
                print(f"   {key}: {value*1000:.1f}ms")
            else:
                print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    return detector

def demo_parameter_comparison():
    """示範不同性能等級的參數比較"""
    print("\n📊 不同硬體性能等級參數比較")
    print("="*50)
    
    # 創建檢測器實例
    detector = HardwarePerformanceDetector()
    
    # 手動設置不同等級並比較參數
    tiers = ["LOW", "MEDIUM", "HIGH", "EXTREME"]
    
    print(f"{'參數名稱':<20} {'LOW':<8} {'MEDIUM':<8} {'HIGH':<8} {'EXTREME':<8}")
    print("-" * 60)
    
    # 收集所有等級的參數
    all_params = {}
    for tier in tiers:
        detector.performance_tier = tier
        all_params[tier] = detector.get_adaptive_parameters()
    
    # 比較關鍵參數
    key_params = [
        ('queue_high_watermark', '{}%'),
        ('queue_low_watermark', '{}%'),
        ('batch_timeout', '{:.1f}ms'),
        ('max_queue_size', '{}'),
        ('fps_check_interval', '{}'),
        ('worker_check_interval', '{:.1f}ms')
    ]
    
    for param_name, format_str in key_params:
        row = f"{param_name:<20}"
        for tier in tiers:
            value = all_params[tier][param_name]
            if param_name.endswith('timeout'):
                formatted_value = format_str.format(value * 1000)
            else:
                formatted_value = format_str.format(value)
            row += f" {formatted_value:<8}"
        print(row)

def demo_adaptive_benefits():
    """說明硬體適應性的好處"""
    print("\n🚀 硬體適應性系統的好處")
    print("="*50)
    
    benefits = [
        "📈 高性能設備自動解除FPS限制",
        "⚡ EXTREME等級: 1ms超時, 100幀隊列, 極致性能",
        "🔄 HIGH等級: 2ms超時, 80幀隊列, 高效處理", 
        "⚖️ MEDIUM等級: 5ms超時, 60幀隊列, 平衡配置",
        "💾 LOW等級: 10ms超時, 40幀隊列, 節省資源",
        "🔍 即時性能監控與動態參數調整",
        "📊 硬體基準測試確保準確分類",
        "🛡️ 自動防止低階硬體過載",
        "🔧 可手動強制重新檢測硬體變更"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print(f"\n💡 關鍵優勢: 同一套程式碼可在從樹莓派到工作站的各種硬體上達到最佳性能")

def demo_performance_scenarios():
    """展示不同硬體場景的處理方式"""
    print("\n🎯 不同硬體場景的處理策略")
    print("="*50)
    
    scenarios = {
        "EXTREME": {
            "硬體範例": "高階工作站 (32核心, 64GB記憶體)",
            "優化策略": "最大隊列容量, 最短超時, 充分利用硬體能力",
            "預期FPS": "50+ FPS (無人工限制)",
            "適用場景": "AI推理服務器, 高畫質影片處理"
        },
        "HIGH": {
            "硬體範例": "遊戲主機 (16核心, 32GB記憶體)", 
            "優化策略": "較大隊列, 短超時, 高效並行處理",
            "預期FPS": "35+ FPS",
            "適用場景": "即時影像處理, 直播串流"
        },
        "MEDIUM": {
            "硬體範例": "一般電腦 (8核心, 16GB記憶體)",
            "優化策略": "平衡配置, 穩定性與性能並重",
            "預期FPS": "25+ FPS", 
            "適用場景": "開發測試, 一般應用"
        },
        "LOW": {
            "硬體範例": "單板電腦 (4核心, 4GB記憶體)",
            "優化策略": "保守配置, 防止記憶體溢出",
            "預期FPS": "15+ FPS",
            "適用場景": "邊緣計算, IoT設備"
        }
    }
    
    for tier, info in scenarios.items():
        print(f"\n{tier} 等級:")
        for key, value in info.items():
            print(f"   {key}: {value}")

def main():
    """主示範函數"""
    print("🌟 MTK Genio 硬體適應性 Pipeline 系統示範")
    print("="*60)
    
    try:
        # 1. 硬體檢測示範
        detector = demo_hardware_detection()
        
        # 2. 參數比較示範
        demo_parameter_comparison()
        
        # 3. 好處說明
        demo_adaptive_benefits()
        
        # 4. 場景示範
        demo_performance_scenarios()
        
        # 5. 實際Pipeline配置示範
        print("\n🔧 實際Pipeline配置示範")
        print("="*50)
        
        # 模擬Pipeline初始化（不實際運行）
        class MockProducer:
            def __init__(self):
                self.mode = "video"
        
        class MockWorkerPool:
            def __init__(self):
                pass
                
        class MockConsumer:
            def __init__(self):
                pass
                
        class MockMonitor:
            def __init__(self):
                pass
        
        # 創建Pipeline實例來展示配置
        mock_producer = MockProducer()
        mock_worker_pool = MockWorkerPool()
        mock_consumer = MockConsumer()
        mock_monitor = MockMonitor()
        
        pipeline = Pipeline(mock_producer, mock_worker_pool, mock_consumer, mock_monitor)
        
        # 顯示硬體配置摘要
        pipeline.print_hardware_summary()
        
        # 展示硬體資訊API
        print("📋 硬體資訊 API 示範:")
        hardware_info = pipeline.get_hardware_info()
        print(f"   獲取到的資訊鍵值: {list(hardware_info.keys())}")
        
        print("\n✅ 硬體適應性系統示範完成!")
        print("   系統已準備好在任何硬體上達到最佳性能 🚀")
        
    except Exception as e:
        print(f"❌ 示範過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

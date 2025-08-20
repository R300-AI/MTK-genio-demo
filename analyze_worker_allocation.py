#!/usr/bin/env python3
"""
Worker分配分析工具
專門用於分析和可視化worker分配問題，找出導致worker數為0的根本原因
"""

import re
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
from typing import List, Dict, Tuple, Optional

class WorkerAllocationAnalyzer:
    """Worker分配分析器"""
    
    def __init__(self, log_file_path: str = "gstreamer_demo.log"):
        self.log_file_path = log_file_path
        self.timeline_data = []  # 時間軸數據
        self.worker_events = []  # Worker事件記錄
        self.queue_states = []   # Queue狀態記錄
        self.zero_worker_periods = []  # Worker為0的時間段
        
    def parse_log_file(self):
        """解析log文件，提取timeline調試信息"""
        print(f"📖 開始解析log文件: {self.log_file_path}")
        
        if not os.path.exists(self.log_file_path):
            print(f"❌ 找不到log文件: {self.log_file_path}")
            return False
            
        timeline_pattern = r'\[TIMELINE-DEBUG\] (.+)'
        timeline_alert_pattern = r'\[TIMELINE-ALERT\] (.+)'
        worker_pattern = r'\[WORKER-DEBUG\] (.+)'
        queue_pattern = r'\[QUEUE-DEBUG\] (.+)'
        
        with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # 提取時間戳
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                timestamp = None
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    except:
                        pass
                
                # 解析Timeline Alert信息（優先級較高）
                alert_match = re.search(timeline_alert_pattern, line)
                if alert_match:
                    self._parse_timeline_entry(alert_match.group(1), timestamp, line_num)
                    continue
                
                # 解析Timeline調試信息
                timeline_match = re.search(timeline_pattern, line)
                if timeline_match:
                    self._parse_timeline_entry(timeline_match.group(1), timestamp, line_num)
                    continue
                
                # 解析Worker調試信息
                worker_match = re.search(worker_pattern, line)
                if worker_match:
                    self._parse_worker_entry(worker_match.group(1), timestamp, line_num)
                
                # 解析Queue調試信息  
                queue_match = re.search(queue_pattern, line)
                if queue_match:
                    self._parse_queue_entry(queue_match.group(1), timestamp, line_num)
        
        print(f"✅ 解析完成: {len(self.timeline_data)} 條時間軸記錄")
        print(f"   Worker事件: {len(self.worker_events)} 條")
        print(f"   Queue狀態: {len(self.queue_states)} 條")
        
        return True
    
    def _parse_timeline_entry(self, entry: str, timestamp: Optional[datetime], line_num: int):
        """解析時間軸條目"""
        try:
            data = {
                'timestamp': timestamp,
                'line_num': line_num,
                'raw': entry
            }
            
            # 檢查是否是TIMELINE-ALERT格式
            if "所有Worker都處於非活動狀態" in entry:
                # 解析格式: "t=0.1s - 所有Worker都處於非活動狀態! InputQueue=0, Workers=2, ActiveWorkers=0"
                time_match = re.search(r't=([0-9.]+)s', entry)
                input_queue_match = re.search(r'InputQueue=(\d+)', entry)
                workers_match = re.search(r'Workers=(\d+)', entry)
                active_workers_match = re.search(r'ActiveWorkers=(\d+)', entry)
                
                if time_match:
                    data['timeline_time'] = float(time_match.group(1))
                if input_queue_match:
                    data['queue'] = {
                        'input_size': int(input_queue_match.group(1)),
                        'input_max': 100,  # 假設
                        'output_size': 0,   # 從alert無法得知
                        'output_max': 100   # 假設
                    }
                if workers_match and active_workers_match:
                    data['workers'] = {
                        'total_count': int(workers_match.group(1)),
                        'active_count': int(active_workers_match.group(1))
                    }
                
                data['is_alert'] = True
                self.timeline_data.append(data)
                return
            
            # 解析標準TIMELINE-DEBUG格式：
            # "t=0.2s | Producer:📸(Frame#1) | InputQ:[0(0%)] | Workers:1/3⚙️💤💤 | OutputQ:[0(0%)] | Consumer:💻(Result#0)"
            
            # 提取時間
            time_match = re.search(r't=([0-9.]+)s', entry)
            if time_match:
                data['timeline_time'] = float(time_match.group(1))
            
            # 提取Producer信息
            producer_match = re.search(r'Producer:(📸|⏸️)\(Frame#(\d+)\)', entry)
            if producer_match:
                data['producer'] = {
                    'status': 'active' if producer_match.group(1) == '📸' else 'inactive',
                    'frame_count': int(producer_match.group(2))
                }
            
            # 提取InputQ信息
            input_queue_match = re.search(r'InputQ:\[(\d+)\((\d+)%\)\]', entry)
            if input_queue_match:
                input_size = int(input_queue_match.group(1))
                input_percent = int(input_queue_match.group(2))
                # 根據百分比反推最大值
                input_max = 100 if input_percent == 0 and input_size == 0 else max(1, int(input_size * 100 / max(1, input_percent)))
                data['input_queue'] = {
                    'size': input_size,
                    'max': input_max,
                    'percent': input_percent
                }
            
            # 提取Workers信息
            workers_match = re.search(r'Workers:(\d+)/(\d+)', entry)
            if workers_match:
                data['workers'] = {
                    'active_count': int(workers_match.group(1)),
                    'total_count': int(workers_match.group(2))
                }
            
            # 提取OutputQ信息
            output_queue_match = re.search(r'OutputQ:\[(\d+)\((\d+)%\)\]', entry)
            if output_queue_match:
                output_size = int(output_queue_match.group(1))
                output_percent = int(output_queue_match.group(2))
                output_max = 100 if output_percent == 0 and output_size == 0 else max(1, int(output_size * 100 / max(1, output_percent)))
                data['output_queue'] = {
                    'size': output_size,
                    'max': output_max,
                    'percent': output_percent
                }
            
            # 提取Consumer信息
            consumer_match = re.search(r'Consumer:(💻|⏹️)\(Result#(\d+)\)', entry)
            if consumer_match:
                data['consumer'] = {
                    'status': 'active' if consumer_match.group(1) == '💻' else 'inactive',
                    'result_count': int(consumer_match.group(2))
                }
            
            self.timeline_data.append(data)
            
        except Exception as e:
            print(f"⚠️  解析時間軸條目失敗 (行{line_num}): {e}")
            print(f"   原始內容: {entry[:100]}...")
    
    def _parse_worker_entry(self, entry: str, timestamp: Optional[datetime], line_num: int):
        """解析Worker事件條目"""
        try:
            self.worker_events.append({
                'timestamp': timestamp,
                'line_num': line_num,
                'content': entry
            })
        except Exception as e:
            print(f"⚠️  解析Worker事件失敗 (行{line_num}): {e}")
    
    def _parse_queue_entry(self, entry: str, timestamp: Optional[datetime], line_num: int):
        """解析Queue狀態條目"""
        try:
            self.queue_states.append({
                'timestamp': timestamp,
                'line_num': line_num,
                'content': entry
            })
        except Exception as e:
            print(f"⚠️  解析Queue狀態失敗 (行{line_num}): {e}")
    
    def find_zero_worker_periods(self):
        """找出worker數為0的時間段"""
        print("\n🔍 尋找worker數為0的時間段...")
        
        zero_periods = []
        current_zero_start = None
        
        for i, data in enumerate(self.timeline_data):
            if 'workers' in data:
                active_workers = data['workers']['active_count']
                
                if active_workers == 0:
                    if current_zero_start is None:
                        current_zero_start = i
                else:
                    if current_zero_start is not None:
                        # 結束一個zero period
                        zero_periods.append({
                            'start_index': current_zero_start,
                            'end_index': i - 1,
                            'duration_snapshots': i - current_zero_start,
                            'start_data': self.timeline_data[current_zero_start],
                            'end_data': self.timeline_data[i - 1] if i > 0 else None
                        })
                        current_zero_start = None
        
        # 處理最後一個未結束的zero period
        if current_zero_start is not None:
            zero_periods.append({
                'start_index': current_zero_start,
                'end_index': len(self.timeline_data) - 1,
                'duration_snapshots': len(self.timeline_data) - current_zero_start,
                'start_data': self.timeline_data[current_zero_start],
                'end_data': self.timeline_data[-1]
            })
        
        self.zero_worker_periods = zero_periods
        
        print(f"✅ 找到 {len(zero_periods)} 個worker為0的時間段")
        for i, period in enumerate(zero_periods):
            print(f"   期間{i+1}: 持續 {period['duration_snapshots']} 個時間快照")
        
        return zero_periods
    
    def analyze_zero_periods(self):
        """分析worker為0期間的詳細情況"""
        if not self.zero_worker_periods:
            print("❌ 沒有找到worker為0的時間段")
            return
        
        print(f"\n📊 分析 {len(self.zero_worker_periods)} 個worker為0時間段:")
        print("="*70)
        
        for i, period in enumerate(self.zero_worker_periods):
            print(f"\n🔎 時間段 {i+1}:")
            print(f"   持續快照數: {period['duration_snapshots']}")
            
            start_data = period['start_data']
            end_data = period['end_data']
            
            print(f"   開始狀態 (行{start_data.get('line_num', '?')}):")
            if 'producer' in start_data:
                prod = start_data['producer']
                print(f"     Producer: {prod['status']}, frame={prod['frame_count']}")
            if 'queue' in start_data:
                queue = start_data['queue']
                print(f"     Queue: input={queue['input_size']}/{queue['input_max']}, "
                      f"output={queue['output_size']}/{queue['output_max']}")
            if 'consumer' in start_data:
                cons = start_data['consumer']
                print(f"     Consumer: {cons['status']}, results={cons['result_count']}")
            
            if end_data:
                print(f"   結束狀態 (行{end_data.get('line_num', '?')}):")
                if 'producer' in end_data:
                    prod = end_data['producer']
                    print(f"     Producer: {prod['status']}, frame={prod['frame_count']}")
                if 'queue' in end_data:
                    queue = end_data['queue']
                    print(f"     Queue: input={queue['input_size']}/{queue['input_max']}, "
                          f"output={queue['output_size']}/{queue['output_max']}")
                if 'consumer' in end_data:
                    cons = end_data['consumer']
                    print(f"     Consumer: {cons['status']}, results={cons['result_count']}")
            
            # 分析此期間的可能原因
            print(f"   🤔 可能原因分析:")
            self._analyze_period_causes(period)
    
    def _analyze_period_causes(self, period: dict):
        """分析特定時間段worker為0的可能原因"""
        start_idx = period['start_index']
        end_idx = period['end_index']
        
        # 檢查期間內的queue狀態變化
        input_queue_sizes = []
        output_queue_sizes = []
        
        for i in range(max(0, start_idx-2), min(len(self.timeline_data), end_idx+3)):
            data = self.timeline_data[i]
            if 'queue' in data:
                input_queue_sizes.append(data['queue']['input_size'])
                output_queue_sizes.append(data['queue']['output_size'])
        
        # 分析可能的原因
        causes = []
        
        if input_queue_sizes and all(size == 0 for size in input_queue_sizes[-3:]):
            causes.append("輸入隊列為空 - Producer可能暫停或結束")
        
        if output_queue_sizes and all(size >= 40 for size in output_queue_sizes[-3:]):
            causes.append("輸出隊列接近滿載 - Consumer處理速度慢")
        
        if len(input_queue_sizes) > 1:
            if input_queue_sizes[-1] > input_queue_sizes[0] * 2:
                causes.append("輸入隊列快速增長 - 可能存在批次同步問題")
        
        if not causes:
            causes.append("需要進一步分析worker內部狀態")
        
        for cause in causes:
            print(f"     • {cause}")
    
    def generate_visual_timeline(self, start_index: int = 0, length: int = 50):
        """生成視覺化時間軸"""
        print(f"\n📈 視覺化時間軸 (從快照{start_index}開始，長度{length}):")
        print("="*80)
        
        end_index = min(start_index + length, len(self.timeline_data))
        
        print("時間軸說明:")
        print("P=Producer活動, I=輸入Queue(數字), W=Worker數量, O=輸出Queue(數字), C=Consumer活動")
        print("符號: ✅=活動, ❌=非活動, 🔴=異常狀態")
        print("-"*80)
        
        for i in range(start_index, end_index):
            data = self.timeline_data[i]
            line_info = f"快照{i:3d}"
            
            # Producer狀態
            if 'producer' in data:
                status = "✅" if data['producer']['status'] == 'active' else "❌"
                line_info += f" P{status}"
            else:
                line_info += " P?"
            
            # 輸入Queue
            if 'queue' in data:
                input_size = data['queue']['input_size']
                if input_size == 0:
                    line_info += " I🔴"
                elif input_size >= 40:
                    line_info += f" I{input_size:2d}⚠️"
                else:
                    line_info += f" I{input_size:2d}"
            else:
                line_info += " I?"
            
            # Worker數量
            if 'workers' in data:
                worker_count = data['workers']['active_count']
                if worker_count == 0:
                    line_info += " W🔴0"
                else:
                    line_info += f" W{worker_count}"
            else:
                line_info += " W?"
            
            # 輸出Queue
            if 'queue' in data:
                output_size = data['queue']['output_size']
                if output_size >= 40:
                    line_info += f" O{output_size:2d}⚠️"
                else:
                    line_info += f" O{output_size:2d}"
            else:
                line_info += " O?"
            
            # Consumer狀態
            if 'consumer' in data:
                status = "✅" if data['consumer']['status'] == 'active' else "❌"
                line_info += f" C{status}"
            else:
                line_info += " C?"
            
            print(line_info)
    
    def generate_summary_report(self):
        """生成總結報告"""
        print(f"\n📋 Worker分配問題總結報告")
        print("="*70)
        
        total_snapshots = len(self.timeline_data)
        zero_worker_snapshots = sum(1 for data in self.timeline_data 
                                   if 'workers' in data and data['workers']['active_count'] == 0)
        
        print(f"總時間快照數: {total_snapshots}")
        print(f"Worker為0的快照數: {zero_worker_snapshots}")
        if total_snapshots > 0:
            zero_percentage = (zero_worker_snapshots / total_snapshots) * 100
            print(f"Worker為0的時間比例: {zero_percentage:.1f}%")
        
        print(f"Worker為0的時間段數: {len(self.zero_worker_periods)}")
        
        if self.zero_worker_periods:
            avg_duration = sum(p['duration_snapshots'] for p in self.zero_worker_periods) / len(self.zero_worker_periods)
            max_duration = max(p['duration_snapshots'] for p in self.zero_worker_periods)
            print(f"平均持續時間: {avg_duration:.1f} 快照")
            print(f"最長持續時間: {max_duration} 快照")
        
        # 統計worker數量分佈
        worker_counts = []
        for data in self.timeline_data:
            if 'workers' in data:
                worker_counts.append(data['workers']['active_count'])
        
        if worker_counts:
            count_distribution = Counter(worker_counts)
            print(f"\nWorker數量分佈:")
            for count in sorted(count_distribution.keys()):
                percentage = (count_distribution[count] / len(worker_counts)) * 100
                print(f"  {count} workers: {count_distribution[count]} 次 ({percentage:.1f}%)")
        
        print("\n🎯 建議:")
        if zero_worker_snapshots > 0:
            print("• 發現worker分配為0的情況，建議優化worker調度算法")
            print("• 檢查Producer和Consumer的同步機制")
            print("• 考慮實現adaptive worker池大小調整")
        else:
            print("• Worker分配正常，無發現嚴重問題")

def main():
    """主函數"""
    print("🔍 MTK Genio Worker分配問題分析工具")
    print("="*60)
    
    log_file = "gstreamer_demo.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    analyzer = WorkerAllocationAnalyzer(log_file)
    
    # 解析log文件
    if not analyzer.parse_log_file():
        return
    
    # 尋找worker為0的時間段
    analyzer.find_zero_worker_periods()
    
    # 分析這些時間段
    analyzer.analyze_zero_periods()
    
    # 生成視覺化時間軸
    if analyzer.timeline_data:
        choice = input(f"\n要顯示視覺化時間軸嗎？ (y/n): ").strip().lower()
        if choice == 'y':
            start_idx = 0
            if analyzer.zero_worker_periods:
                # 顯示第一個problem period周圍的時間軸
                first_problem = analyzer.zero_worker_periods[0]
                start_idx = max(0, first_problem['start_index'] - 10)
            
            analyzer.generate_visual_timeline(start_idx, 30)
    
    # 生成總結報告
    analyzer.generate_summary_report()
    
    print(f"\n✅ 分析完成！")
    print(f"建議運行: python demo_timeline_debug.py 來生成更多調試數據")

if __name__ == "__main__":
    main()

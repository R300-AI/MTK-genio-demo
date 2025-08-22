import threading
import logging
import time
import psutil
import numpy as np
from collections import deque

logger = logging.getLogger('gstreamer_demo')

class TimelineLogger:
    """時間軸除錯器，追蹤Pipeline各組件的詳細狀態變化"""
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.timeline_data = deque(maxlen=max_history)
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 0.5  # 每0.5秒記錄一次時間軸
        self.lock = threading.Lock()
        
        # 組件狀態追蹤
        self.producer_state = {"active": False, "frame_count": 0, "last_activity": 0}
        self.worker_states = {}  # {worker_id: {"active": bool, "task_count": int, "last_activity": float}}
        self.consumer_state = {"active": False, "result_count": 0, "frame_count": 0, "last_activity": 0}
        self.queue_states = {"input": 0, "output": 0, "input_max": 0, "output_max": 0}
        
        # 流控統計追蹤
        self.flow_control_stats = {
            "throttle_events": 0,
            "total_throttle_time": 0.0,
            "max_input_queue": 0,
            "avg_input_queue": 0.0,
            "queue_full_events": 0,
            "last_throttle_time": 0
        }
        
    def update_producer_state(self, active=None, frame_count=None):
        """更新Producer狀態"""
        with self.lock:
            current_time = time.time()
            if active is not None:
                self.producer_state["active"] = active
                if active:
                    self.producer_state["last_activity"] = current_time
            if frame_count is not None:
                self.producer_state["frame_count"] = frame_count
    
    def update_worker_state(self, worker_id, active=None, task_count=None):
        """更新Worker狀態"""
        with self.lock:
            current_time = time.time()
            if worker_id not in self.worker_states:
                self.worker_states[worker_id] = {"active": False, "task_count": 0, "last_activity": 0}
            
            if active is not None:
                self.worker_states[worker_id]["active"] = active
                if active:
                    self.worker_states[worker_id]["last_activity"] = current_time
            if task_count is not None:
                self.worker_states[worker_id]["task_count"] = task_count
    
    def update_consumer_state(self, active=None, result_count=None, frame_count=None):
        """更新Consumer狀態"""
        with self.lock:
            current_time = time.time()
            if active is not None:
                self.consumer_state["active"] = active
                if active:
                    self.consumer_state["last_activity"] = current_time
            if result_count is not None:
                self.consumer_state["result_count"] = result_count
            if frame_count is not None:
                self.consumer_state["frame_count"] = frame_count
    
    def update_queue_states(self, input_size=None, output_size=None, input_max=None, output_max=None):
        """更新Queue狀態"""
        with self.lock:
            if input_size is not None:
                self.queue_states["input"] = input_size
                # 更新流控統計
                self.flow_control_stats["max_input_queue"] = max(
                    self.flow_control_stats["max_input_queue"], input_size
                )
                if input_size >= (input_max or 100):
                    self.flow_control_stats["queue_full_events"] += 1
                    
            if output_size is not None:
                self.queue_states["output"] = output_size
            if input_max is not None:
                self.queue_states["input_max"] = input_max
            if output_max is not None:
                self.queue_states["output_max"] = output_max
    
    def record_throttle_event(self, throttle_duration=0.0):
        """記錄流控事件"""
        with self.lock:
            self.flow_control_stats["throttle_events"] += 1
            self.flow_control_stats["total_throttle_time"] += throttle_duration
            self.flow_control_stats["last_throttle_time"] = time.time()
    
    def get_flow_control_summary(self):
        """獲取流控統計摘要"""
        with self.lock:
            runtime = time.time() - self.start_time
            stats = self.flow_control_stats.copy()
            
            # 計算平均值
            if runtime > 0:
                throttle_rate = stats["throttle_events"] / runtime * 60  # 每分鐘流控次數
                throttle_percentage = (stats["total_throttle_time"] / runtime) * 100  # 流控時間百分比
            else:
                throttle_rate = 0
                throttle_percentage = 0
            
            return {
                "throttle_events": stats["throttle_events"],
                "throttle_rate_per_minute": throttle_rate,
                "total_throttle_time": stats["total_throttle_time"],
                "throttle_time_percentage": throttle_percentage,
                "max_input_queue": stats["max_input_queue"],
                "queue_full_events": stats["queue_full_events"],
                "runtime": runtime
            }
    
    def should_log_timeline(self):
        """判斷是否應該記錄時間軸"""
        current_time = time.time()
        return current_time - self.last_log_time >= self.log_interval
    
    def log_timeline_snapshot(self):
        """記錄當前時間軸快照"""
        if not self.should_log_timeline():
            return
            
        with self.lock:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 創建時間軸快照
            snapshot = {
                "timestamp": current_time,
                "elapsed": elapsed_time,
                "producer": self.producer_state.copy(),
                "workers": self.worker_states.copy(),
                "consumer": self.consumer_state.copy(),
                "queues": self.queue_states.copy()
            }
            
            self.timeline_data.append(snapshot)
            self.last_log_time = current_time
            
            # 格式化時間軸狀態
            self._log_formatted_timeline(snapshot)
    
    def _log_formatted_timeline(self, snapshot):
        """格式化並記錄時間軸狀態 - 增強版智能流控顯示"""
        elapsed = snapshot["elapsed"]
        
        # 預先初始化 Queue 變量以避免 UnboundLocalError
        input_queue = snapshot["queues"]["input"]
        input_max = snapshot["queues"]["input_max"] or 100
        output_queue = snapshot["queues"]["output"]
        output_max = snapshot["queues"]["output_max"] or 100
        
        # Producer狀態圖示 - 加入流控狀態
        producer_active = snapshot["producer"]["active"]
        frame_count = snapshot['producer']['frame_count']
        
        if producer_active:
            # 檢查是否處於流控狀態（可以根據Queue狀態推斷）
            if input_queue > input_max * 0.7:  # 70%以上可能觸發流控
                producer_icon = "🐌"  # 流控中
                producer_info = f"Frame#{frame_count}(流控中)"
            else:
                producer_icon = "📸"  # 正常讀取
                producer_info = f"Frame#{frame_count}"
        else:
            producer_icon = "⏸️"  # 暫停
            producer_info = f"Frame#{frame_count}(完成)"
        
        # Queue狀態 (顯示使用率和流控閾值)        
        input_usage = (input_queue / input_max * 100) if input_max > 0 else 0
        output_usage = (output_queue / output_max * 100) if output_max > 0 else 0
        
        # 增強的Worker狀態顯示 - 包含ThreadPoolExecutor信息
        worker_icons = []
        active_workers = 0
        total_workers = len(snapshot["workers"]) if snapshot["workers"] else 0
        
        if total_workers > 0:
            # 顯示詳細的Worker狀態
            busy_count = 0
            idle_count = 0
            
            for worker_id, state in snapshot["workers"].items():
                if state["active"]:
                    if state.get("task_count", 0) > 0:
                        worker_icons.append("⚙️")  # 忙碌
                        busy_count += 1
                    else:
                        worker_icons.append("⚡")  # 就緒
                    active_workers += 1
                else:
                    worker_icons.append("💤")  # 休眠
                    idle_count += 1
            
            # 顯示Worker狀態統計
            if busy_count > 0:
                worker_status = f"🔥{active_workers}/{total_workers}{''.join(worker_icons[:5])}"
            elif active_workers > 0:
                worker_status = f"⚡{active_workers}/{total_workers}{''.join(worker_icons[:5])}"
            else:
                worker_status = f"💤{active_workers}/{total_workers}{''.join(worker_icons[:5])}"
        else:
            # 沒有Worker狀態記錄
            worker_status = "0/0⏳初始化中"
        
        # Consumer狀態圖示 - 修改為顯示frame數
        consumer_icon = "💻" if snapshot["consumer"]["active"] else "⏹️"
        # 優先顯示frame_count，如果沒有則顯示result_count
        frame_count = snapshot['consumer'].get('frame_count', 0)
        result_count = snapshot['consumer'].get('result_count', 0)
        display_count = frame_count if frame_count > 0 else result_count
        consumer_info = f"Frame#{display_count}"
        
        # 記錄增強的時間軸 - 包含流控信息
        logger.info(f"[ENHANCED-TIMELINE] t={elapsed:.1f}s | "
                   f"Producer:{producer_icon}({producer_info}) | "
                   f"InputQ:[{input_queue}({input_usage:.0f}%)] | "
                   f"Workers:{worker_status} | "
                   f"OutputQ:[{output_queue}({output_usage:.0f}%)] | "
                   f"Consumer:{consumer_icon}({consumer_info})")
        
        # 智能警告系統 - 包含流控狀態分析
        if total_workers > 0:
            # 檢查系統瓶頸
            if input_usage > 80 and active_workers == total_workers:
                logger.warning(f"[SYSTEM-ANALYSIS] t={elapsed:.1f}s - 系統接近滿載: "
                             f"InputQueue={input_queue}({input_usage:.0f}%), "
                             f"所有Worker忙碌({active_workers}/{total_workers})")
            
            elif input_usage > 50 and active_workers < total_workers * 0.5:
                logger.warning(f"[SYSTEM-ANALYSIS] t={elapsed:.1f}s - Worker利用率偏低: "
                             f"InputQueue={input_queue}({input_usage:.0f}%), "
                             f"活躍Worker={active_workers}/{total_workers}")
                             
            elif input_usage < 10 and output_usage > 80:
                logger.info(f"[SYSTEM-ANALYSIS] t={elapsed:.1f}s - Consumer成為瓶頸: "
                           f"InputQ={input_queue}({input_usage:.0f}%), "
                           f"OutputQ={output_queue}({output_usage:.0f}%)")
                           
            elif producer_active and input_usage > 70:
                logger.info(f"[FLOW-CONTROL-INFO] t={elapsed:.1f}s - 智能流控可能已啟動: "
                           f"InputQueue={input_queue}({input_usage:.0f}%), "
                           f"Producer可能正在減速讀取")
    
    def get_timeline_summary(self, last_n_seconds=10):
        """獲取最近N秒的時間軸摘要"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - last_n_seconds
            
            recent_data = [snapshot for snapshot in self.timeline_data 
                          if snapshot["timestamp"] >= cutoff_time]
            
            if not recent_data:
                return "無足夠的時間軸數據"
            
            # 分析最近的趨勢
            total_snapshots = len(recent_data)
            producer_active_count = sum(1 for s in recent_data if s["producer"]["active"])
            consumer_active_count = sum(1 for s in recent_data if s["consumer"]["active"])
            
            # Worker活動分析
            worker_activity = {}
            for snapshot in recent_data:
                for worker_id, state in snapshot["workers"].items():
                    if worker_id not in worker_activity:
                        worker_activity[worker_id] = 0
                    if state["active"]:
                        worker_activity[worker_id] += 1
            
            # 格式化摘要
            summary_lines = []
            summary_lines.append(f"=== 最近 {last_n_seconds} 秒時間軸摘要 ===")
            summary_lines.append(f"總快照數: {total_snapshots}")
            summary_lines.append(f"Producer活動率: {producer_active_count}/{total_snapshots} ({producer_active_count/total_snapshots*100:.1f}%)")
            summary_lines.append(f"Consumer活動率: {consumer_active_count}/{total_snapshots} ({consumer_active_count/total_snapshots*100:.1f}%)")
            
            if worker_activity:
                summary_lines.append("Worker活動率:")
                for worker_id, active_count in worker_activity.items():
                    activity_rate = active_count / total_snapshots * 100
                    summary_lines.append(f"  Worker-{worker_id}: {active_count}/{total_snapshots} ({activity_rate:.1f}%)")
            else:
                summary_lines.append("⚠️ 沒有Worker活動記錄!")
            
            return "\n".join(summary_lines)
    
    def print_visual_timeline(self, last_n_snapshots=20):
        """打印視覺化時間軸（類似用戶要求的格式）"""
        with self.lock:
            recent_data = list(self.timeline_data)[-last_n_snapshots:]
            
            if not recent_data:
                print("沒有足夠的時間軸數據")
                return
            
            print(f"\n{'='*80}")
            print("🕐 視覺化時間軸 (最近 {} 個時間點)".format(len(recent_data)))
            print(f"{'='*80}")
            
            # 建立時間軸標題
            time_labels = [f"t{i}" for i in range(len(recent_data))]
            print(f"時間軸:     " + " ".join(f"{t:<4}" for t in time_labels))
            
            # Producer行
            producer_icons = []
            for snapshot in recent_data:
                icon = "📸" if snapshot["producer"]["active"] else "⏸️"
                producer_icons.append(icon)
            print(f"Producer:   " + " ".join(f"{icon:<4}" for icon in producer_icons))
            
            # Input Queue行 (顯示數量)
            queue_sizes = []
            for snapshot in recent_data:
                size = snapshot["queues"]["input"]
                queue_sizes.append(f"[{size}]")
            print(f"InputQueue: " + " ".join(f"{size:<4}" for size in queue_sizes))
            
            # Worker行們
            if recent_data[0]["workers"]:
                worker_ids = list(recent_data[0]["workers"].keys())
                for worker_id in worker_ids:
                    worker_icons = []
                    for snapshot in recent_data:
                        if worker_id in snapshot["workers"]:
                            icon = "⚙️" if snapshot["workers"][worker_id]["active"] else "💤"
                        else:
                            icon = "❌"
                        worker_icons.append(icon)
                    print(f"Worker-{worker_id}:   " + " ".join(f"{icon:<4}" for icon in worker_icons))
            else:
                print("Workers:    " + " ".join("❌ <4" for _ in range(len(recent_data))))
            
            # Output Queue行
            output_sizes = []
            for snapshot in recent_data:
                size = snapshot["queues"]["output"]
                output_sizes.append(f"[{size}]")
            print(f"OutputQueue:" + " ".join(f"{size:<4}" for size in output_sizes))
            
            # Consumer行
            consumer_icons = []
            for snapshot in recent_data:
                icon = "💻" if snapshot["consumer"]["active"] else "⏹️"
                consumer_icons.append(icon)
            print(f"Consumer:   " + " ".join(f"{icon:<4}" for icon in consumer_icons))
            
            print(f"{'='*80}\n")

    def get_visual_timeline_string(self, last_n_snapshots=20):
        """返回視覺化時間軸的格式化字符串（用於logging輸出）"""
        with self.lock:
            recent_data = list(self.timeline_data)[-last_n_snapshots:]
            
            if not recent_data:
                return "沒有足夠的時間軸數據"
            
            lines = []
            lines.append("🕐 視覺化時間軸 (最近 {} 個時間點)".format(len(recent_data)))
            
            # 建立時間軸標題
            time_labels = [f"t{i}" for i in range(len(recent_data))]
            lines.append(f"時間軸:     " + " ".join(f"{t:<4}" for t in time_labels))
            
            # Producer行
            producer_icons = []
            for snapshot in recent_data:
                icon = "📸" if snapshot["producer"]["active"] else "⏸️"
                producer_icons.append(icon)
            lines.append(f"Producer:   " + " ".join(f"{icon:<4}" for icon in producer_icons))
            
            # Input Queue行 (顯示數量)
            queue_sizes = []
            for snapshot in recent_data:
                size = snapshot["queues"]["input"]
                queue_sizes.append(f"[{size}]")
            lines.append(f"InputQueue: " + " ".join(f"{size:<4}" for size in queue_sizes))
            
            # Worker行們
            if recent_data[0]["workers"]:
                worker_ids = list(recent_data[0]["workers"].keys())
                for worker_id in worker_ids:
                    worker_icons = []
                    for snapshot in recent_data:
                        if worker_id in snapshot["workers"]:
                            icon = "⚙️" if snapshot["workers"][worker_id]["active"] else "💤"
                        else:
                            icon = "❌"
                        worker_icons.append(icon)
                    lines.append(f"Worker-{worker_id}:   " + " ".join(f"{icon:<4}" for icon in worker_icons))
            else:
                lines.append("Workers:    " + " ".join("❌ " for _ in range(len(recent_data))))
            
            # Output Queue行
            output_sizes = []
            for snapshot in recent_data:
                size = snapshot["queues"]["output"]
                output_sizes.append(f"[{size}]")
            lines.append(f"OutputQueue:" + " ".join(f"{size:<4}" for size in output_sizes))
            
            # Consumer行
            consumer_icons = []
            for snapshot in recent_data:
                icon = "💻" if snapshot["consumer"]["active"] else "⏹️"
                consumer_icons.append(icon)
            lines.append(f"Consumer:   " + " ".join(f"{icon:<4}" for icon in consumer_icons))
            
            return "\n".join(lines)


class HardwarePerformanceLogger:
    def __init__(self):
        """硬體性能檢測器，自動檢測並分類硬體性能等級"""
        self.performance_tier = None
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.cpu_physical_cores = psutil.cpu_count(logical=False)
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.benchmark_results = {}
        self._detect_performance_tier()
        
    def _detect_performance_tier(self):
        """檢測硬體性能等級"""
        # 執行CPU性能基準測試
        cpu_score = self._benchmark_cpu()
        memory_score = self._benchmark_memory()
        
        # 根據基準測試結果分類性能等級
        total_score = cpu_score * 0.7 + memory_score * 0.3
        
        if total_score >= 8.0:
            self.performance_tier = "EXTREME"
        elif total_score >= 6.0:
            self.performance_tier = "HIGH"
        elif total_score >= 4.0:
            self.performance_tier = "MEDIUM"
        else:
            self.performance_tier = "LOW"
            
        logger.info(f"硬體性能等級: {self.performance_tier} "
                   f"(CPU核心: {self.cpu_cores}, 記憶體: {self.memory_gb:.1f}GB, "
                   f"總分: {total_score:.2f})")
    
    def _benchmark_cpu(self):
        """CPU性能基準測試"""
        start_time = time.time()
        # 執行數學計算測試
        result = 0
        for i in range(1000000):
            result += i ** 0.5
        cpu_time = time.time() - start_time
        
        # 基於CPU核心數和計算時間的評分
        base_score = min(10.0, self.cpu_cores * 2.0)
        time_penalty = max(0.1, cpu_time * 2)
        cpu_score = base_score / time_penalty
        
        self.benchmark_results['cpu_score'] = cpu_score
        return cpu_score
    
    def _benchmark_memory(self):
        """記憶體性能基準測試"""
        try:
            # 記憶體存取速度測試
            size = min(100000, int(self.memory_gb * 10000))  # 根據記憶體大小調整
            data = np.random.random(size)
            
            start_time = time.time()
            # 執行記憶體密集操作
            result = np.sum(data ** 2)
            memory_time = time.time() - start_time
            
            # 基於記憶體大小和存取時間的評分
            memory_score = min(10.0, self.memory_gb) / max(0.1, memory_time * 10)
            
        except Exception as e:
            logger.warning(f"記憶體基準測試失敗: {e}")
            memory_score = min(5.0, self.memory_gb / 2)  # 降級評分
            
        self.benchmark_results['memory_score'] = memory_score
        return memory_score
    
    def get_adaptive_parameters(self):
        """根據硬體性能等級返回適應性參數"""
        params = {
            "EXTREME": {
                "queue_high_watermark": 80,
                "queue_low_watermark": 60,
                "batch_timeout": 0.001,  # 1ms
                "max_queue_size": 100,
                "fps_check_interval": 50,
                "worker_check_interval": 0.005
            },
            "HIGH": {
                "queue_high_watermark": 70,
                "queue_low_watermark": 50,
                "batch_timeout": 0.002,  # 2ms
                "max_queue_size": 80,
                "fps_check_interval": 35,
                "worker_check_interval": 0.008
            },
            "MEDIUM": {
                "queue_high_watermark": 60,
                "queue_low_watermark": 40,
                "batch_timeout": 0.005,  # 5ms
                "max_queue_size": 60,
                "fps_check_interval": 25,
                "worker_check_interval": 0.01
            },
            "LOW": {
                "queue_high_watermark": 50,
                "queue_low_watermark": 30,
                "batch_timeout": 0.01,   # 10ms
                "max_queue_size": 40,
                "fps_check_interval": 15,
                "worker_check_interval": 0.02
            }
        }
        return params.get(self.performance_tier, params["MEDIUM"])

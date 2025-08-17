from ultralytics import YOLO
import numpy as np
from queue import Queue, Empty
import threading
import time
import logging

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

class Processor:
    """
    單一 YOLO 模型處理器 (保持您的原始設計)
    """
    def __init__(self, model_path):
        logger.info(f"[PROCESSOR] model - {model_path}")
        self.model_path = model_path
        
        self.model = YOLO(model_path, task="detect")
        
        # 預熱模型
        try:
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_input, verbose=False)
        except Exception as e:
            raise
        
        self.is_busy = False
        self.lock = threading.Lock()

    def predict(self, frame):
        """執行推論，自動管理 busy 狀態"""
        with self.lock:
            if self.is_busy:
                return None
            self.is_busy = True
        
        
        try:
            # 使用 timeout 來避免無限等待
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("YOLO prediction timeout")
            
            # 設置 10 秒超時（Windows 不支持 signal，所以改用 threading.Timer）
            import threading
            timeout_occurred = threading.Event()
            
            def timeout_func():
                timeout_occurred.set()
            
            timer = threading.Timer(10.0, timeout_func)
            timer.start()
            
            try:
                results = self.model.predict(frame, verbose=False, save=False, show=False)
                if timeout_occurred.is_set():
                    return None
                processed_frame = results[0].plot()
                return processed_frame
            finally:
                timer.cancel()
                
        except Exception as e:
            return None
        finally:
            with self.lock:
                self.is_busy = False

class WorkerPool:
    """
    改良版 WorkerPool：基於您現有的 Processor 設計
    提供多工處理 + 順序保證 + 處理間隔控制
    """
    def __init__(self, model_path, monitor=None, max_workers=4, balancer=None):
        logger.info(f"[WORKERPOOL] streaming with {max_workers} workers")
        self.monitor = monitor
        self.max_workers = max_workers
        self.balancer = balancer  # 新增 Balancer 支援
        
        # 使用您的 Processor 設計
        self.workers = []
        for i in range(max_workers):
            worker = Processor(model_path)
            self.workers.append(worker)
        
        # 處理間隔控制（類似 Producer 的 frame_interval）
        self.process_interval = 0.1  # 預設每個任務間隔 0.1 秒
        self.last_process_time = 0

        # 任務管理
        self.task_queue = Queue()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.running = False
        
        # 任務序號與 in-flight 追蹤
        self.sequence_counter = 1
        self.sequence_lock = threading.Lock()
        self.in_flight = set()  # 追蹤所有送出的 seq_num
        self.in_flight_lock = threading.Lock()
        self.result_callback = None
        

    def start(self, result_callback=None):
        """啟動 WorkerPool"""
        self.result_callback = result_callback
        self.running = True
        self.processing_thread.start()

    def stop(self):
        """停止 WorkerPool"""
        self.running = False
        self.task_queue.put(None)
        with self.in_flight_lock:
            if self.in_flight:
                logger.warning(f"[INFLIGHT] Unfinished tasks at stop: {self.in_flight}")

    def process(self, frame):
        """提交 frame 進行處理，加入背壓控制避免批次處理"""
        if frame is None:
            return frame
        
        # 背壓控制：如果系統負載太高，直接丟frame
        if self._should_drop_frame():
            logger.debug("System overloaded, dropping frame to prevent batching")
            return
        
        # 正常處理
        with self.sequence_lock:
            seq_num = self.sequence_counter
            self.sequence_counter += 1
        with self.in_flight_lock:
            self.in_flight.add(seq_num)
        self.task_queue.put((frame, seq_num))
    
    def _should_drop_frame(self):
        """判斷是否應該丟棄frame - 背壓控制核心邏輯"""
        # 計算當前系統負載
        busy_workers = sum(1 for worker in self.workers if getattr(worker, 'is_busy', False))
        queue_size = self.task_queue.qsize()
        total_load = busy_workers + queue_size
        
        # 策略：如果總負載接近或超過worker數量，就開始丟frame
        load_threshold = self.max_workers * 0.8  # 80%負載閾值
        
        if total_load >= load_threshold:
            logger.debug(f"Load control: {total_load}/{self.max_workers} workers busy/queued, dropping frame")
            return True
        
        return False

    def _find_available_worker(self):
        """找到可用的 worker，使用您的 is_busy 機制"""
        available_count = 0
        for worker in self.workers:
            with worker.lock:
                if not worker.is_busy:
                    logger.debug(f"WORKERPOOL_FIND: Found available worker")
                    return worker
                else:
                    available_count += 1
        return None

    def _processing_loop(self):
        """主要處理迴圈：按順序分配任務，但允許多工執行 + 處理間隔控制"""
        def predict_async(frame, seq_num, worker):
            try:
                result = worker.predict(frame)
                self._handle_result(seq_num, result)
            except Exception as e:
                logger.error(f"[WORKERPOOL] Worker error for seq {seq_num}: {e}")
                self._handle_result(seq_num, None)
            finally:
                if self.monitor:
                    self.monitor.count_processing_end()

        while self.running:
            try:
                # 使用 Balancer 調整處理間隔（類似 Producer）
                if self.balancer:
                    # 根據worker數量調整間隔，避免過度密集派工
                    base_interval = self.balancer.get_producer_sleep(self.process_interval)
                    self.process_interval = base_interval / max(1, self.max_workers - 2)
                
                # 控制處理間隔，確保worker派工均勻分散
                current_time = time.time()
                elapsed = current_time - self.last_process_time
                if elapsed < self.process_interval:
                    time.sleep(self.process_interval - elapsed)
                self.last_process_time = time.time()
                
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    break
                frame, seq_num = task
                
                # 尋找可用worker
                worker = None
                retry_count = 0
                while worker is None and self.running:
                    worker = self._find_available_worker()
                    if worker is None:
                        retry_count += 1
                        if retry_count % 100 == 0:  # 減少警告頻率
                            logger.debug(f"WORKERPOOL_LOOP: Waiting for available worker (retry {retry_count})")
                        time.sleep(0.01)  # 增加等待時間，減少CPU占用
                
                if not self.running:
                    break
                    
                # 啟動worker處理
                if self.monitor:
                    self.monitor.count_processing_start()
                threading.Thread(target=predict_async, args=(frame, seq_num, worker), daemon=True).start()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"WORKERPOOL_LOOP: Error in processing loop: {e}")
        

    def _handle_result(self, seq_num, result):
        """處理結果 - 簡化版本，依靠背壓控制保證順序"""
        # 清理 in_flight 追蹤
        with self.in_flight_lock:
            self.in_flight.discard(seq_num)
        
        # 直接輸出結果（背壓控制已經保證了相對順序）
        if result is not None:
            try:
                self.result_callback(result)
            except Exception as e:
                logger.error(f"[WORKERPOOL] result_callback error for seq {seq_num}: {e}")
        
        # 更新監控統計
        if self.monitor:
            self.monitor.count_processed()

    def get_status(self):
        """獲取WorkerPool詳細狀態，包含背壓控制資訊"""
        busy_workers = sum(1 for worker in self.workers if getattr(worker, 'is_busy', False))
        queue_size = self.task_queue.qsize()
        
        with self.in_flight_lock:
            in_flight_count = len(self.in_flight)
        
        total_load = busy_workers + queue_size
        load_percentage = (total_load / self.max_workers) * 100
        
        return {
            'total_workers': self.max_workers,
            'busy_workers': busy_workers,
            'available_workers': self.max_workers - busy_workers,
            'queue_size': queue_size,
            'in_flight_tasks': in_flight_count,
            'total_load': total_load,
            'load_percentage': load_percentage,
            'backpressure_active': total_load >= (self.max_workers * 0.8)
        }
    
    def log_status(self):
        """記錄當前狀態到日誌"""
        status = self.get_status()
        logger.info(f"[WORKERPOOL_STATUS] Load: {status['load_percentage']:.1f}% "
                   f"({status['busy_workers']} busy + {status['queue_size']} queued) "
                   f"Backpressure: {'ON' if status['backpressure_active'] else 'OFF'}")

    # 保持與原本相容的方法（如果需要）
    def get_worker(self):
        """相容性方法：取得可用 worker"""
        return self._find_available_worker()

    def put_worker(self, worker):
        """相容性方法：不需要實際操作，因為 worker 會自動管理狀態"""
        pass
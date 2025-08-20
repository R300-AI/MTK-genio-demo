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
        self.model = YOLO(model_path)
        # 預熱模型
        try:
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_input, verbose=False, stream=True)
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
            timeout_occurred = threading.Event()
            def timeout_func():
                timeout_occurred.set()
                logger.warning(f"[PROCESSOR] Prediction timeout (10s) occurred!")

            timer = threading.Timer(10.0, timeout_func)
            timer.start()
            try:
                results = self.model.predict(frame, verbose=False, save=False, show=False)
                
                if timeout_occurred.is_set():
                    logger.warning(f"[PROCESSOR] Prediction timed out, returning None")
                    return None
                
                processed_frame = results[0].plot(boxes=False)
                return processed_frame
            finally:
                timer.cancel()
        except Exception as e:
            import traceback
            logger.error(f"[PROCESSOR] Prediction failed: {e}")
            logger.error(f"[PROCESSOR] Traceback: {traceback.format_exc()}")
            return None
        finally:
            with self.lock:
                self.is_busy = False

class WorkerPool:
    """
    提供多工處理 + 順序保證 + 處理間隔控制
    """
    def __init__(self, model_path, monitor=None, max_workers=4, balancer=None, mode='camera'):
        logger.info(f"[WORKERPOOL] streaming with {max_workers} workers, mode={mode}")
        self.monitor = monitor
        self.max_workers = max_workers
        self.balancer = balancer  # 新增 Balancer 支援
        self.mode = mode

        # 使用您的 Processor 設計
        self.workers = []
        for i in range(max_workers):
            worker = Processor(model_path)
            self.workers.append(worker)

        # 任務管理（queue buffer 設定 maxlen，video mode 需要更大的 buffer）
        from collections import deque
        if mode == 'video':
            # video mode: 大 buffer，確保所有幀都能處理
            self.queue_maxlen = 200  # 足夠處理大多數短片
        else:
            # camera mode: 小 buffer，保持即時性
            self.queue_maxlen = max_workers * 2
        self.task_queue = deque(maxlen=self.queue_maxlen)
        self.task_queue_lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.running = False

        # 任務序號與 in-flight 追蹤
        self.sequence_counter = 1
        self.sequence_lock = threading.Lock()
        self.in_flight = set()  # 追蹤所有送出的 seq_num
        self.in_flight_lock = threading.Lock()
        self.result_callback = None
        
        # Video mode 專用的順序保證機制
        if mode == 'video':
            self.result_buffer = {}  # {seq_num: result}
            self.next_output_seq = 1
            self.result_buffer_lock = threading.Lock()
            
        # 添加處理速率追蹤變數
        self.processed_counter = 0
        self.last_processing_fps_time = time.time()
        self.processing_fps_check_interval = 20  # 每20幀檢查一次處理fps
        self.queue_size_log_counter = 0  # queue size記錄計數器


    def start(self, result_callback=None):
        """啟動 WorkerPool"""
        self.result_callback = result_callback
        self.running = True
        self.processing_thread.start()

    def stop(self):
        """停止 WorkerPool"""
        self.running = False
        with self.task_queue_lock:
            self.task_queue.append(None)
        with self.in_flight_lock:
            if self.in_flight:
                logger.warning(f"[INFLIGHT] Unfinished tasks at stop: {self.in_flight}")

    def process(self, frame_tuple):
        """提交 (frame, timestamp) 進行處理，queue buffer 滿時自動丟棄最舊 frame"""
        process_start_time = time.time()
        
        # 支援 (frame, timestamp) 或單一 frame
        if isinstance(frame_tuple, tuple) and len(frame_tuple) == 2:
            frame, timestamp = frame_tuple
        else:
            frame = frame_tuple
            timestamp = time.time()

        if frame is None:
            return frame

        # 僅 camera mode 才允許 drop frame
        if self.mode == 'camera' and self._should_drop_frame():
            # ...移除開發階段 debug log...
            return

        # video mode 一律強制入列
        with self.sequence_lock:
            seq_num = self.sequence_counter
            self.sequence_counter += 1
        with self.in_flight_lock:
            self.in_flight.add(seq_num)
        with self.task_queue_lock:
            if len(self.task_queue) == self.queue_maxlen:
                pass  # queue 滿時可在此處理丟棄或警告
            self.task_queue.append(((frame, timestamp), seq_num))
            queue_size = len(self.task_queue)
            
        # 每50幀記錄一次queue狀態的DEBUG資訊
        self.queue_size_log_counter += 1
        if self.queue_size_log_counter % 50 == 0:
            process_time = time.time() - process_start_time
            logger.debug(f"[DEBUG] [WORKERPOOL] Mode={self.mode}, Queue_size={queue_size}/{self.queue_maxlen}, "
                        f"Process_submit_time={process_time:.4f}s, Task#{seq_num}")
    
    def _should_drop_frame(self):
        """判斷是否應該丟棄frame - 背壓控制核心邏輯"""
        # 計算當前系統負載
        busy_workers = sum(1 for worker in self.workers if getattr(worker, 'is_busy', False))
        queue_size = len(self.task_queue)
        total_load = busy_workers + queue_size

        # 策略：如果總負載接近或超過worker數量，就開始丟frame
        load_threshold = self.max_workers * 0.8  # 80%負載閾值

        if total_load >= load_threshold:
            # ...移除開發階段 debug log...
            return True

        return False

    def _find_available_worker(self):
        """使用 round-robin 策略找到可用的 worker，避免總是使用前面的 worker"""
        if not hasattr(self, '_last_worker_index'):
            self._last_worker_index = 0
        
        # 從上次使用的 worker 下一個位置開始搜索
        start_index = (self._last_worker_index + 1) % len(self.workers)
        
        # 搜索兩輪：第一輪從 start_index 開始，第二輪從頭開始
        for round_num in range(2):
            start = start_index if round_num == 0 else 0
            end = len(self.workers) if round_num == 0 else start_index
            
            for i in range(start, end):
                worker = self.workers[i]
                # 使用 trylock 避免阻塞，如果無法獲得鎖說明 worker 很忙
                if worker.lock.acquire(blocking=False):
                    try:
                        if not worker.is_busy:
                            self._last_worker_index = i
                            # ...移除開發階段 debug log...
                            return worker
                    finally:
                        worker.lock.release()
        return None

    def _processing_loop(self):
        """主要處理迴圈：按順序分配任務，但允許多工執行，並根據 timestamp 丟棄過時 frame"""
        # 設定最大允許延遲（秒），超過則丟棄 frame
        MAX_LATENESS = 0.5  # 可依需求調整

        def predict_async(frame, timestamp, seq_num, worker):
            # video mode 不檢查過時，camera mode 才檢查
            now = time.time()
            if self.mode == 'camera' and now - timestamp > MAX_LATENESS:
                # ...移除開發階段 debug log...
                self._handle_result(seq_num, None)
                if self.monitor:
                    self.monitor.count_processing_end()
                return
            try:
                result = worker.predict(frame)
                # ...移除開發階段 debug/info log...
                self._handle_result(seq_num, result)
            except Exception as e:
                logger.error(f"[WORKERPOOL] Worker error for seq {seq_num}: {e}")
                self._handle_result(seq_num, None)
            finally:
                if self.monitor:
                    self.monitor.count_processing_end()

        import time
        while self.running:
            try:
                with self.task_queue_lock:
                    if not self.task_queue:
                        time.sleep(0.01)
                        continue
                    task = self.task_queue.popleft()
                    
                if task is None:
                    # ...移除開發階段 debug/info log...
                    break
                (frame, timestamp), seq_num = task

                # 尋找可用worker
                worker = None
                retry_count = 0
                while worker is None and self.running:
                    worker = self._find_available_worker()
                    if worker is None:
                        retry_count += 1
                        if retry_count % 100 == 0:
                            pass  # 可在此處加警告或統計
                        time.sleep(0.01)

                if not self.running:
                    break

                # 啟動worker處理
                if self.monitor:
                    self.monitor.count_processing_start()
                threading.Thread(target=predict_async, args=(frame, timestamp, seq_num, worker), daemon=True).start()
            except Exception as e:
                logger.error(f"WORKERPOOL_LOOP: Error in processing loop: {e}")

    def _handle_result(self, seq_num, result):
        """處理結果 - 增強版本，video mode 保證順序"""
        result_start_time = time.time()
        
        # 清理 in_flight 追蹤
        with self.in_flight_lock:
            self.in_flight.discard(seq_num)
        
        self.processed_counter += 1
        
        if self.mode == 'video':
            # video mode: 需要按順序輸出
            with self.result_buffer_lock:
                self.result_buffer[seq_num] = result
                
                # 按順序輸出可以輸出的結果
                while self.next_output_seq in self.result_buffer:
                    buffered_result = self.result_buffer.pop(self.next_output_seq)
                    # video mode: 即使結果是 None 也要回調，讓 Consumer 知道這幀處理完了
                    try:
                        if buffered_result is not None:
                            self.result_callback(buffered_result)
                        else:
                            pass  # result is None 時不做任何事
                    except Exception as e:
                        logger.error(f"[WORKERPOOL] result_callback error for seq {self.next_output_seq}: {e}")
                    self.next_output_seq += 1
        else:
            # camera mode: 直接輸出（保持原有邏輯）
            if result is not None:
                try:
                    self.result_callback(result)
                except Exception as e:
                    logger.error(f"[WORKERPOOL] result_callback error for seq {seq_num}: {e}")
            else:
                pass  # result is None 時不做任何事
                
        # 定期記錄處理fps的DEBUG資訊
        if self.processed_counter % self.processing_fps_check_interval == 0:
            current_time = time.time()
            actual_interval = (current_time - self.last_processing_fps_time) / self.processing_fps_check_interval
            actual_processing_fps = 1.0 / actual_interval if actual_interval > 0 else 0
            result_time = current_time - result_start_time
            
            logger.debug(f"[DEBUG] [WORKERPOOL] Mode={self.mode}, Processed#{self.processed_counter}, "
                        f"Actual_Processing_FPS={actual_processing_fps:.2f}, Result_handle_time={result_time:.4f}s, "
                        f"Result_success={'Yes' if result is not None else 'No'}, Seq#{seq_num}")
            self.last_processing_fps_time = current_time
        
        # 更新監控統計
        if self.monitor:
            self.monitor.count_processed()

    def get_status(self):
        """獲取WorkerPool詳細狀態，包含背壓控制資訊"""
        busy_workers = sum(1 for worker in self.workers if getattr(worker, 'is_busy', False))
        queue_size = len(self.task_queue)

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
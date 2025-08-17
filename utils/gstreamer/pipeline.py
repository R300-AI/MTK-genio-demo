import threading
import logging
import time
from queue import Queue, Empty

# 設定 logging 配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gstreamer_demo.log', mode='w', encoding='utf-8')
    ]
)

# 建立專用的 logger
logger = logging.getLogger('gstreamer_demo')

class Pipeline:
    def __init__(self, producer, worker_pool, consumer, monitor=None, balancer=None):
        logger.info("Creating pipeline...")
        self.producer = producer
        self.worker_pool = worker_pool
        self.consumer = consumer
        self.monitor = monitor
        self.balancer = balancer  # 新增 Balancer
    
        self.input_queue = Queue(maxsize=50)  # 增加緩衝區大小
        self.output_queue = Queue(maxsize=50)  # 增加緩衝區大小
        self.running = False

        # 設定 Balancer 與 Monitor 的連結
        if self.balancer and self.monitor:
            self.balancer.set_monitor(self.monitor)
            self.balancer.set_producer(self.producer)

    def run(self):
        self.running = True

        # 定義結果處理回調函數：將 WorkerPool 的結果放入 output_queue
        def result_handler(result):
            if result is not None:
                try:
                    self.output_queue.put(result, timeout=1.0)
                except Exception as e:
                    logger.error(f"PIPELINE_CALLBACK: Failed to queue result: {e}")

        # 啟動 WorkerPool 並設定結果回調
        self.worker_pool.start(result_handler)

        producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        
        worker_thread.start()
        producer_thread.start()
        consumer_thread.start()
        
        try:
            producer_thread.join()
            
            # 檢查 Producer 模式
            producer_mode = getattr(self.producer, 'mode', 'camera')
            
            if producer_mode == "video":
                # 影片模式：等待所有幀都被處理和消費完畢
                logger.info("PIPELINE_RUN: Video mode - waiting for all frames to be processed and consumed")
                
                # 等待 WorkerPool 處理完所有剩餘的幀
                self._wait_for_processing_completion()
                
                # 停止 WorkerPool 並等待它完成
                self.worker_pool.stop()
                self.output_queue.put(None)
            else:
                # 攝影機模式：立即停止（原來的行為）
                logger.info("PIPELINE_RUN: Camera mode - stopping immediately")
                self.input_queue.put(None)
                
                worker_thread.join()
                self.output_queue.put(None)
            
            consumer_thread.join()
        except Exception as e:
            logger.error(f"PIPELINE_RUN: Error during execution: {e}")
        finally:
            self.running = False
            self.worker_pool.stop()

    def _producer_loop(self):
        """讀取 frame 並放入 input_queue"""
        frame_count = 0
        
        try:
            for frame in self.producer:
                if not self.running:
                    break
                
                frame_count += 1
                
                try:
                    self.input_queue.put(frame, timeout=1.0)
                    # Monitor 會自動在 Producer 中記錄，這裡不需要額外記錄
                except Exception as e:
                    logger.error(f"PRODUCER: Failed to queue frame {frame_count}: {e}")
                    break
        except StopIteration:
            logger.info("PRODUCER: StopIteration caught - end of frames")
        except Exception as e:
            logger.error(f"PRODUCER: Error in producer loop: {e}")
        
        logger.info(f"PRODUCER: Producer loop ended, total frames processed: {frame_count}")

    def _worker_loop(self):
        """工作者迴圈：從 input_queue 取 frame 並交給 WorkerPool 處理"""
        processed_count = 0
        
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                if frame is None:
                    break
                
                processed_count += 1
                
                # 交給 WorkerPool 處理，結果會透過回調函數自動放入 output_queue
                self.worker_pool.process(frame)
                # Monitor 會自動在 WorkerPool 中記錄處理狀態，這裡不需要額外記錄
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"WORKER_LOOP: Error processing frame {processed_count}: {e}")


    def _consumer_loop(self):
        """從 output_queue 取結果並顯示"""
        consumed_count = 0
        
        while self.running:
            try:
                result = self.output_queue.get(timeout=0.1)
                if result is None:
                    break
                
                consumed_count += 1

                try:
                    self.consumer.consume(result)
                    # Monitor 會自動在 Consumer 中記錄，這裡不需要額外記錄
                except Exception as e:
                    logger.error(f"CONSUMER: Error consuming result {consumed_count}: {e}")
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"CONSUMER: Error in consumer loop: {e}")
    
    def _wait_for_processing_completion(self):
        # 停止接受新的輸入
        self.input_queue.put(None)
        
        # 添加超時機制，避免無限等待
        timeout = 30  # 30秒超時
        start_wait_time = time.time()
        stable_count = 0  # 計算狀態穩定的次數
        last_state = None
        
        # 等待所有幀完成整個流水線（產生->處理->消費）
        while True:
            if self.monitor:
                with self.monitor.lock:
                    produced = self.monitor.frame_count
                    processed = self.monitor.processed_count
                    consumed = self.monitor.consumed_count
                    processing = self.monitor.processing
                    
                    current_state = (produced, processed, consumed, processing)
                    should_finish = False
                    
                    if produced > 0 and processed >= produced and consumed >= produced and processing == 0:
                        should_finish = True

                    elif produced > 0 and processing == 0 and (processed >= produced - 1) and (consumed >= produced - 1):
                        should_finish = True
                    elif processing == 0 and current_state == last_state:
                        stable_count += 1
                        if stable_count >= 10:
                            should_finish = True
                    else:
                        stable_count = 0
                    if time.time() - start_wait_time > timeout:
                        should_finish = True
                    if should_finish:
                        break

                    last_state = current_state
            
            time.sleep(0.1)  # 短暫等待
        

"""
================================================================================
🏗️ Pipeline 架構設計 2025.08.23 (更新版)
================================================================================

Pipeline類採用繼承架構，支援Video模式（完整性優先）和Camera模式（實時性優先）。

� Frame ID 追蹤整合 (2025.08.23)：
Pipeline 現在支援完整的 frame_id 追蹤，從 Producer 產生開始到 Consumer 顯示結束，
確保每個幀都能在整個處理管道中保持可追蹤性，完美支援多線程並行處理環境。

�🎯 核心組件：
┌─────────────┬──────────────────┬─────────────────────────────────────────┐
│ 📸 Producer │ 幀生產與推送     │ Video:確保無丟幀 / Camera:協商式流控    │
│ ⚙️ Worker   │ 幀處理與推理     │ Video:硬體適應性 / Camera:背壓檢測      │
│ 🖥️ Consumer │ 結果顯示與輸出   │ Video:完整保證 / Camera:實時優化        │
└─────────────┴──────────────────┴─────────────────────────────────────────┘

📊 資料流向 (整合架構更新版)：
    Producer ──> ThreadPoolExecutor ──[output_queue]──> Consumer
    (Frame+ID)     (直接處理)            (Results+ID)     (追蹤顯示)

🎯 繼承關係：
                    BasePipeline (抽象基類)
                    ├── 通用初始化和硬體檢測
                    ├── 統一執行流程模板  
                    ├── Frame ID 追蹤機制 🆕
                    └── 抽象方法定義
                           │
            ┌──────────────┴──────────────┐
            │                             │
    VideoPipeline                  CameraPipeline
    (完整性優先)                   (實時性優先)

📊 職責分配：
┌─────────────────┬──────────────────┬─────────────────┬─────────────────┐
│   功能類別      │   BasePipeline    │  VideoPipeline  │ CameraPipeline  │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ 🚀 初始化管理   │ ✅ 硬體檢測配置   │ ✅ 無丟幀參數   │ ✅ 實時性參數   │
│ 🎯 主控制流     │ ✅ run()模板方法  │ 🔹 繼承使用     │ 🔹 繼承使用     │
│ 📸 Producer邏輯 │ 🔹 抽象方法       │ ✅ 確保無丟幀   │ ✅ 協商式流控   │
│ ⚙️ Worker邏輯   │ 🔹 抽象方法       │ ✅ 硬體適應性   │ ✅ 背壓檢測     │
│ 🖥️ Consumer邏輯 │ 🔹 抽象方法       │ ✅ 完整顯示     │ ✅ 智能丟幀     │
│ 🏁 完成處理     │ 🔹 抽象方法       │ ✅ 等待完整     │ ✅ 快速停止     │
│ 📊 性能監控     │ ✅ 基礎監控       │ ✅ 參數自適應   │ ✅ 能力協商     │
│ 🛠️ 工具方法     │ ✅ 硬體資訊/調試  │ 🔹 繼承使用     │ 🔹 繼承使用     │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┘

🔧 核心特性：
• 硬體適應性：自動檢測硬體(LOW/MEDIUM/HIGH/EXTREME)並調整參數
• Timeline調試：追蹤Producer/Worker/Consumer狀態，生成視覺化時間軸
• 能力協商：Camera模式組件自主報告能力，協商FPS目標

🛠️ 開發提示：
• 新增模式：在各loop方法中添加分支，實現對應*_loop_[模式]()方法
• 調試工具：enable_intensive_debug_logging(), print_debug_timeline()
• 注意事項：Video重完整性可能影響實時性，Camera重實時性可能丟幀
"""

import threading
import time
import signal
import logging
import queue
import os
from queue import Queue
from abc import ABC, abstractmethod
from .metric import TimelineLogger, HardwarePerformanceLogger

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gstreamer_demo.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('gstreamer_demo')


class BasePipeline(ABC):
    """
    抽象Pipeline基類 - Template Method Pattern
    
    提供通用的初始化、硬體檢測和執行模板，
    子類實現具體的Producer/Worker/Consumer邏輯
    """
    
    def __init__(self, producer, worker_pool, consumer):
        """通用初始化：硬體檢測、Queue配置、性能監控"""
        logger.info("📋 步驟 1/4: 🚀 組件類型檢測與註冊...")
        logger.info(f"� Producer類型: {type(producer).__name__}")
        logger.info(f"� WorkerPool類型: {type(worker_pool).__name__}")
        logger.info(f"� Consumer類型: {type(consumer).__name__}")
        
        self.producer = producer
        self.worker_pool = worker_pool
        self.consumer = consumer
        
        # 從WorkerPool獲取硬體資訊（避免重複檢測）
        logger.info("📋 步驟 2/4: 🔧 從WorkerPool取得硬體資訊與適應性參數...")
        if hasattr(self.worker_pool, '_performance_level'):
            hardware_tier = getattr(self.worker_pool, '_performance_level', '中等性能')
            cpu_cores = getattr(self.worker_pool, '_cpu_cores', 4)
            memory_gb = getattr(self.worker_pool, '_memory_gb', 8.0)
            logger.info(f"📊 硬體資訊來源: WorkerPool")
            logger.info(f"📊 硬體性能等級: {hardware_tier} (CPU核心: {cpu_cores}, 記憶體: {memory_gb:.1f}GB)")
        else:
            # 如果 WorkerPool 沒有硬體檢測，使用預設值
            hardware_tier = '中等性能'
            logger.warning("⚠️ WorkerPool未提供硬體資訊，使用預設等級: 中等性能")
        
        # 根據硬體等級生成Pipeline專用的適應性參數
        self.adaptive_params = self._generate_pipeline_adaptive_params(hardware_tier)
        logger.info(f"📊 Pipeline適應性參數: {self.adaptive_params}")
        
        # Timeline調試工具
        logger.info("📋 步驟 3/4: ⚙️ 初始化Timeline調試工具與性能監控...")
        self.timeline_debugger = TimelineLogger()
        
        # 🎯 整合架構：直接使用ThreadPoolExecutor，移除INPUT_QUEUE
        logger.info("📋 步驟 4/4: 🎯 整合架構初始化 - 直接使用ThreadPoolExecutor...")
        max_queue_size = self.adaptive_params["max_queue_size"]
        logger.info(f"🔧 整合模式: 直接提交到ThreadPoolExecutor，只使用OUTPUT_QUEUE")
        logger.info(f"🔧 配置OUTPUT_QUEUE - 最大大小: {max_queue_size}")
        
        self.output_queue = Queue(maxsize=max_queue_size)
        self.running = False
        
        # 🎯 整合架構：初始化專用的ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(4, os.cpu_count() or 4)  # 預設4個worker或CPU核心數
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="Pipeline-Direct"
        )
        logger.info(f"🚀 Pipeline專用ThreadPoolExecutor已創建：{max_workers}個worker")
        
        logger.info("✅ 整合架構Queue創建完成")
        
        # 性能監控變數
        self.pipeline_frame_counter = 0
        self.last_pipeline_fps_time = time.time()
        self.pipeline_fps_check_interval = self.adaptive_params["fps_check_interval"]
        logger.info(f"⏱️ 性能監控間隔設定: {self.pipeline_fps_check_interval}秒")
        
        # Producer狀態追蹤
        self.producer_finished = False
        self.producer_last_activity = time.time()
        self.producer_activity_timeout = 2.0
        
        # 性能監控歷史數據
        self.performance_history = []
        self.performance_history_max_length = 100
        logger.info(f"📊 性能歷史數據緩存: 最大{self.performance_history_max_length}筆記錄")
        
        # 信號處理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 模式特定初始化
        logger.info("📋 步驟 4/4: 🎯 執行模式特定初始化...")
        self._mode_specific_init()
    
    @abstractmethod
    def _mode_specific_init(self):
        """模式特定的初始化（子類實現）"""
        pass
    
    def _generate_pipeline_adaptive_params(self, hardware_tier):
        """根據硬體等級生成Pipeline專用的適應性參數 - 簡化版"""
        params_map = {
            "高性能": {
                "max_queue_size": 80,
                "fps_check_interval": 35,
                "batch_timeout": 0.002,
                "queue_high_watermark": 70,
                "queue_low_watermark": 50
            },
            "中等性能": {
                "max_queue_size": 60,
                "fps_check_interval": 25,
                "batch_timeout": 0.005,
                "queue_high_watermark": 60,
                "queue_low_watermark": 40
            },
            "基本性能": {
                "max_queue_size": 40,
                "fps_check_interval": 15,
                "batch_timeout": 0.01,
                "queue_high_watermark": 50,
                "queue_low_watermark": 30
            },
            "未知": {
                "max_queue_size": 50,
                "fps_check_interval": 20,
                "batch_timeout": 0.008,
                "queue_high_watermark": 55,
                "queue_low_watermark": 35
            }
        }
        
        return params_map.get(hardware_tier, params_map["中等性能"])
        return params_map.get(hardware_tier, params_map["MEDIUM"])
    
    def _signal_handler(self, signum, frame):
        """處理系統終止信號"""
        logger.info(f"收到終止信號 {signum}")
        self.running = False
    
    def _start_timeline_logging(self):
        """啟動時間軸除錯logging線程"""
        def timeline_logger():
            while self.running:
                try:
                    self._update_timeline_states()
                    self.timeline_debugger.log_timeline_snapshot()
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Timeline logger error: {e}")
        
        self.timeline_thread = threading.Thread(target=timeline_logger, daemon=True)
        self.timeline_thread.start()
    
    def _update_timeline_states(self):
        """更新時間軸狀態信息 - 整合架構版"""
        try:
            # 🎯 整合架構：監控ThreadPoolExecutor work_queue
            thread_pool_queue_size = 0
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                if hasattr(self.thread_pool, '_work_queue'):
                    thread_pool_queue_size = self.thread_pool._work_queue.qsize()
            
            # 更新Queue狀態 - 使用ThreadPool work_queue作為input_size
            self.timeline_debugger.update_queue_states(
                input_size=thread_pool_queue_size,  # 🎯 監控真實的任務隊列
                output_size=self.output_queue.qsize()
            )
            
            # 增強版Worker狀態追蹤 - 支援ThreadPoolExecutor
            if hasattr(self.worker_pool, 'executor') and self.worker_pool.executor is not None:
                executor = self.worker_pool.executor
                max_workers = self.worker_pool.worker_pool_config.max_workers
                
                # 獲取詳細的執行狀態
                pending_tasks = getattr(self.worker_pool, 'pending_tasks', 0)
                
                # 檢查ThreadPoolExecutor內部工作隊列
                internal_queue_size = 0
                if hasattr(executor, '_work_queue'):
                    internal_queue_size = executor._work_queue.qsize()
                
                # 檢查活躍線程數量
                active_threads = 0
                if hasattr(executor, '_threads'):
                    active_threads = len([t for t in executor._threads if t.is_alive()])
                
                # 估算活躍Worker數量
                # 活躍 = min(pending_tasks + internal_queue_size, max_workers)
                total_workload = pending_tasks + internal_queue_size
                estimated_active = min(total_workload, max_workers) if total_workload > 0 else 0
                
                # 更新每個Worker的狀態
                for i in range(max_workers):
                    worker_id = f"W{i+1}"
                    is_active = i < estimated_active
                    task_count = 1 if is_active else 0
                    self.timeline_debugger.update_worker_state(worker_id, active=is_active, task_count=task_count)
                
                # 記錄詳細的Worker狀態調試信息
                logger.debug(f"[ENHANCED-WORKER-TRACKING] "
                           f"pending_tasks={pending_tasks}, "
                           f"internal_queue={internal_queue_size}, "
                           f"active_threads={active_threads}/{max_workers}, "
                           f"estimated_active={estimated_active}")
            
            elif hasattr(self.worker_pool, 'worker_pool_config'):
                # 如果ThreadPoolExecutor尚未啟動，創建空白Worker狀態
                max_workers = self.worker_pool.worker_pool_config.max_workers
                for i in range(max_workers):
                    worker_id = f"W{i+1}"
                    self.timeline_debugger.update_worker_state(worker_id, active=False, task_count=0)
                logger.debug(f"[WORKER-TRACKING] ThreadPoolExecutor未啟動，設置{max_workers}個Worker為非活躍狀態")
            
        except Exception as e:
            logger.debug(f"Timeline state update error: {e}")
    
    def run(self):
        """Template Method - 統一執行流程模板"""
        self.running = True
        logger.info(" ---------------------------------------------------------")
        logger.info("🚀 開始執行")
        self._start_timeline_logging()
        
        logger.info(f"📊 Pipeline啟動前佇列狀態: 整合架構 - 直接使用ThreadPoolExecutor，output_queue={self.output_queue.qsize()}")
        def result_handler(result):
            if result is not None:
                try:
                    # 🔍 檢查結果是否是多個對象的容器
                    if isinstance(result, (list, tuple)):
                        for idx, item in enumerate(result):
                            queue_size_before = self.output_queue.qsize()
                            self.output_queue.put(item, timeout=1.0)
                            queue_size_after = self.output_queue.qsize()
                            
                    elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                        for idx, item in enumerate(result):
                            queue_size_before = self.output_queue.qsize()
                            self.output_queue.put(item, timeout=1.0)
                            queue_size_after = self.output_queue.qsize()
                            
                    else:
                        # 單一結果
                        queue_size_before = self.output_queue.qsize()
                        self.output_queue.put(result, timeout=1.0)
                        queue_size_after = self.output_queue.qsize()

                    self.timeline_debugger.update_consumer_state(active=True)

                except Exception as e:
                    logger.error(f"❌ [OUTPUT_QUEUE_PUT] 加入失敗: {str(e)} (當前大小: {self.output_queue.qsize()}/{self.output_queue.maxsize})")
            else:
                logger.warning("⚠️ WorkerPool返回了空結果 (None)")

        self.worker_pool.start(result_handler)
        self.consumer.start()

        producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        # 🎯 整合架構：不再需要worker_thread，ThreadPoolExecutor直接處理
        # worker_thread = threading.Thread(target=self._worker_loop, daemon=True)  
        consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)

        producer_thread.start() 
        consumer_thread.start()
        
        try:
            producer_thread.join()

            # 模式特定的完成處理
            logger.info(f"🏁 執行{self.__class__.__name__}完成處理...")
            self._handle_completion()
            consumer_thread.join()

            logger.info("✅ 執行緒已完成")
            logger.info("🛑 停止顯示...")
            self.consumer.stop()
            
            # 打印最終時間軸摘要
            logger.info("📊 ============================================================")
            logger.info("📊 最終性能摘要與清理")
            logger.info("📊 ============================================================")
            timeline_summary = self.timeline_debugger.get_timeline_summary(last_n_seconds=30)
            logger.info(f"📈 Timeline摘要: {timeline_summary}")
            
            # 輸出視覺化時間軸到 log
            visual_timeline = self.timeline_debugger.get_visual_timeline_string(last_n_snapshots=15)
            for line in visual_timeline.split('\n'):
                if line.strip():  # 只輸出非空行
                    logger.info(f"📊 {line}")
            
            # 輸出流控統計摘要
            flow_stats = self.timeline_debugger.get_flow_control_summary()
            logger.info("🚦 ============== 智能流控統計摘要 ==============")
            logger.info(f"🚦 流控事件總數: {flow_stats['throttle_events']}")
            logger.info(f"🚦 流控頻率: {flow_stats['throttle_rate_per_minute']:.1f} 次/分鐘")
            logger.info(f"🚦 總流控時間: {flow_stats['total_throttle_time']:.2f} 秒")
            logger.info(f"🚦 流控時間占比: {flow_stats['throttle_time_percentage']:.2f}%")
            logger.info(f"🚦 最大ThreadPool利用率: {flow_stats['max_input_queue']}")
            logger.info(f"🚦 Queue滿載次數: {flow_stats['queue_full_events']}")
            logger.info(f"🚦 總運行時間: {flow_stats['runtime']:.2f} 秒")
            logger.info("🚦 =============================================")
            
        except Exception as e:
            logger.error(f"❌ PIPELINE_RUN: Error during execution: {e}")
        finally:
            logger.info("🧹 執行清理程序...")
            self.running = False
            self.worker_pool.stop()
            
            # 🎯 整合架構：清理專用ThreadPoolExecutor
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                logger.info("🧹 關閉Pipeline專用ThreadPoolExecutor...")
                self.thread_pool.shutdown(wait=True)
                logger.info("✅ ThreadPoolExecutor已關閉")
                
            if hasattr(self, 'timeline_thread'):
                self.timeline_thread.join(timeout=1.0)
            logger.info("✅ Pipeline執行完成!")
            logger.info("📊 ============================================================")
    
    @abstractmethod
    def _producer_loop(self):
        """Producer邏輯（子類實現）"""
        pass
    
    @abstractmethod
    def _consumer_loop(self):
        """Consumer邏輯（子類實現）"""
        pass
    
    @abstractmethod
    def _handle_completion(self):
        """完成處理（子類實現）"""
        pass
    
    def _collect_recent_performance_metrics(self):
        """收集最近的性能指標用於適應性調整 - 整合架構版"""
        current_time = time.time()
        
        # 🎯 整合架構：收集ThreadPoolExecutor隊列狀態
        thread_pool_utilization = 0.0
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            if hasattr(self.thread_pool, '_work_queue') and hasattr(self.thread_pool, '_max_workers'):
                work_queue_size = self.thread_pool._work_queue.qsize()
                max_workers = self.thread_pool._max_workers
                # 以工作線程的10倍作為滿載基準
                thread_pool_utilization = work_queue_size / (max_workers * 10.0)
                
        output_queue_utilization = self.output_queue.qsize() / self.output_queue.maxsize
        
        # 收集worker狀態
        active_workers = getattr(self.worker_pool, 'active_workers', 0)
        total_workers = getattr(self.worker_pool, 'num_workers', 1)
        worker_utilization = active_workers / total_workers if total_workers > 0 else 0
        
        # 計算當前FPS
        if hasattr(self, 'last_fps_calculation_time') and hasattr(self, 'last_frame_count'):
            time_diff = current_time - self.last_fps_calculation_time
            frame_diff = self.pipeline_frame_counter - self.last_frame_count
            current_fps = frame_diff / time_diff if time_diff > 0 else 0
        else:
            current_fps = 0
        
        # 更新FPS計算基準
        self.last_fps_calculation_time = current_time
        self.last_frame_count = self.pipeline_frame_counter
        
        # 記錄性能數據 - 整合架構版
        performance_metrics = {
            'timestamp': current_time,
            'thread_pool_util': thread_pool_utilization,  # 🎯 使用ThreadPoolExecutor利用率
            'output_queue_util': output_queue_utilization,
            'worker_utilization': worker_utilization,
            'current_fps': current_fps,
            'active_workers': active_workers,
            'total_workers': total_workers
        }
        
        self.performance_history.append(performance_metrics)
        
        # 限制歷史數據長度
        if len(self.performance_history) > self.performance_history_max_length:
            self.performance_history.pop(0)
        
        return performance_metrics


class VideoPipeline(BasePipeline):
    """
    Video模式Pipeline - 完整性優先
    
    特點：
    - 確保無丟幀處理
    - 硬體適應性調整
    - 等待所有幀處理完成
    """
    
    def _mode_specific_init(self):
        """Video模式特定初始化"""
        # Queue預載機制參數
        self.queue_preload_enabled = True
        max_queue_size = self.adaptive_params["max_queue_size"]
        self.min_queue_depth = max(3, max_queue_size // 10)
        self.preload_batch_size = max(2, max_queue_size // 20)
        self.last_preload_check_time = time.time()
        self.preload_check_interval = 0.1
        
        logger.info(f"🔧 Video模式參數配置:")
        logger.info(f"  └─ Queue預載: {'啟用' if self.queue_preload_enabled else '停用'}")
        logger.info(f"  └─ 最小Queue深度: {self.min_queue_depth}")
        logger.info(f"  └─ 預載批次大小: {self.preload_batch_size}")
        logger.info(f"  └─ 預載檢查間隔: {self.preload_check_interval}秒")
        logger.info("✅ Pipeline初始化完成")
    
    def _producer_loop(self):
        """Video模式Producer - 智能流控確保無丟幀"""
        frame_count = 0
        last_adjustment_time = time.time()
        batch_timeout = self.adaptive_params["batch_timeout"]
        # 預載相關變數
        frame_buffer = []
        max_buffer_size = self.preload_batch_size * 2
        self.timeline_debugger.update_producer_state(active=True, frame_count=0)
        
        try:
            for frame in self.producer:
                if not self.running:
                    logger.warning("⚠️ Producer收到停止信號，中斷幀讀取")
                    break
                
                # 🎯 智能流控：檢查系統處理能力
                if self._should_throttle_reading():
                    throttle_wait = self._calculate_throttle_delay()
                    logger.debug(f"🐌 [FLOW-CONTROL] WorkerPool負載過高，暫停讀取 {throttle_wait*1000:.0f}ms")
                    
                    # 記錄流控事件到Timeline
                    self.timeline_debugger.record_throttle_event(throttle_wait)
                    
                    time.sleep(throttle_wait)
                    continue
                
                frame_buffer.append(frame)
                frame_count += 1

                # 批次處理並預載 - 使用智能流控
                if len(frame_buffer) >= self.preload_batch_size:
                    logger.debug(f"📦 執行智能批次處理: {len(frame_buffer)} 幀")
                    success_count = self._process_frame_batch_with_flow_control(frame_buffer, batch_timeout)
                    logger.debug(f"📦 批次處理完成: {success_count}/{len(frame_buffer)} 幀成功放入隊列")
                    frame_buffer = []
                
                # 更新Producer狀態
                self.producer_last_activity = time.time()
                self.timeline_debugger.update_producer_state(active=True, frame_count=frame_count)
                
                # 定期性能調整
                current_time = time.time()
                if current_time - last_adjustment_time > self.pipeline_fps_check_interval:
                    metrics = self._collect_recent_performance_metrics()
                    logger.debug(f"📊 性能指標更新: FPS={metrics['current_fps']:.2f}, "
                               f"Thread Pool={metrics['thread_pool_util']:.2f}, "
                               f"Worker利用率={metrics['worker_utilization']:.2f}")
                    self._adjust_parameters_if_needed(metrics)
                    last_adjustment_time = current_time
            
            # 處理剩餘frames - 使用智能流控
            if frame_buffer:
                logger.info(f"📦 處理剩餘的 {len(frame_buffer)} 幀")
                success_count = self._process_frame_batch_with_flow_control(frame_buffer, batch_timeout)
                logger.info(f"📦 最終批次處理完成: {success_count}/{len(frame_buffer)} 幀成功放入隊列")
            
        except StopIteration:
            logger.info("🏁 Producer迭代完成 (StopIteration)")
        except Exception as e:
            logger.error(f"❌ Producer循環發生錯誤: {e}")
        finally:
            self.producer_finished = True
            self.timeline_debugger.update_producer_state(active=False, frame_count=frame_count)
            logger.info(f"✅ VIDEO PRODUCER 完成: 總共處理 {frame_count} 幀")
    
    def _process_frame_batch_video(self, frame_batch, timeout):
        """🎯 整合架構：Video模式統一使用流控批次處理"""
        # 使用整合架構的流控方法處理所有frame批次
        return self._process_frame_batch_with_flow_control(frame_batch, timeout)

    def _should_throttle_reading(self):
        """🎯 智能流控：檢查是否應該放慢讀取速度 - 整合架構監控ThreadPoolExecutor"""
        try:
            # 🎯 整合架構：直接監控ThreadPoolExecutor
            
            # 1. 檢查ThreadPoolExecutor的work_queue
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                if hasattr(self.thread_pool, '_work_queue'):
                    work_queue_size = self.thread_pool._work_queue.qsize()
                    max_workers = self.thread_pool._max_workers if hasattr(self.thread_pool, '_max_workers') else 2
                    
                    # 閾值：10倍於worker數量
                    work_queue_threshold = max_workers * 10
                    if work_queue_size > work_queue_threshold:
                        logger.debug(f"🐌 [THROTTLE] ThreadPool work_queue過載: {work_queue_size} > {work_queue_threshold}")
                        return True
            
            # 2. 檢查WorkerPool的pending任務
            if hasattr(self.worker_pool, 'executor') and self.worker_pool.executor is not None:
                executor = self.worker_pool.executor
                pending_tasks = getattr(self.worker_pool, 'pending_tasks', 0)
                max_workers = getattr(self.worker_pool.worker_pool_config, 'max_workers', 2)
                
                # 如果pending任務超過worker數量的15倍，則需要流控
                max_pending_threshold = max_workers * 15
                if pending_tasks > max_pending_threshold:
                    logger.debug(f"🐌 [THROTTLE] WorkerPool過載: pending_tasks={pending_tasks} > {max_pending_threshold}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"⚠️ [THROTTLE] 檢查流控狀態時發生錯誤: {e}")
            return False
    
    def _calculate_throttle_delay(self):
        """🎯 計算流控延遲時間 - 整合架構基於ThreadPoolExecutor負載"""
        try:
            # 基礎延遲：50ms
            base_delay = 0.05
            
            # 🎯 整合架構：根據ThreadPoolExecutor work_queue負載調整
            work_queue_multiplier = 1.0
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                if hasattr(self.thread_pool, '_work_queue'):
                    work_queue_size = self.thread_pool._work_queue.qsize()
                    max_workers = self.thread_pool._max_workers if hasattr(self.thread_pool, '_max_workers') else 2
                    
                    # 工作隊列負載：正常化到0-1
                    work_load = work_queue_size / max(max_workers * 10.0, 1)  # 10倍worker為滿載
                    work_queue_multiplier = 1.0 + work_load * 2.0  # 最多3倍基礎延遲
            
            # 根據WorkerPool負載調整
            worker_multiplier = 1.0
            if hasattr(self.worker_pool, 'pending_tasks'):
                pending_tasks = getattr(self.worker_pool, 'pending_tasks', 0)
                max_workers = getattr(self.worker_pool.worker_pool_config, 'max_workers', 2)
                if max_workers > 0:
                    worker_load = pending_tasks / (max_workers * 10.0)  # 正常化到0-1
                    worker_multiplier = 1.0 + min(worker_load * 3.0, 3.0)  # 最多4倍
            
            # 計算最終延遲
            final_delay = base_delay * work_queue_multiplier * worker_multiplier
            return min(final_delay, 1.0)  # 最長1秒延遲
            
        except Exception as e:
            logger.debug(f"⚠️ [THROTTLE] 計算延遲時發生錯誤: {e}")
            return 0.1  # 預設100ms延遲

    def _process_frame_task(self, frame):
        """🎯 整合架構：ThreadPoolExecutor執行的任務方法"""
        try:
            # 🎯 關鍵修正：直接獲取Future結果，而非嵌套Future
            future = self.worker_pool.submit(frame)
            
            if future is None:
                # 背壓控制丟棄了任務
                logger.warning(f"⚠️ [TASK_DROPPED] 任務被背壓控制丟棄")
                return None
                
            # 等待WorkerPool任務完成並獲取結果
            result = future.result(timeout=self.worker_pool.processor_config.inference_timeout)
            return result
            
        except Exception as e:
            logger.error(f"❌ [TASK_ERROR] ThreadPoolExecutor任務處理失敗: {e}")
            return None

    def _process_frame_batch_with_flow_control(self, frame_batch, timeout):
        """🎯 使用流控機制處理frame批次 - 整合架構直接提交到ThreadPoolExecutor"""
        success_count = 0
        
        for i, frame in enumerate(frame_batch):
            # 在每個幀處理前檢查流控
            if self._should_throttle_reading():
                throttle_wait = self._calculate_throttle_delay()
                logger.debug(f"🐌 [BATCH-THROTTLE] 第{i+1}幀前暫停 {throttle_wait*1000:.0f}ms")
                
                # 記錄流控事件
                self.timeline_debugger.record_throttle_event(throttle_wait)
                
                time.sleep(throttle_wait)
            
            try:
                # 提取frame_id用於日誌顯示
                frame_id = frame.get('frame_id', i) if isinstance(frame, dict) else i
                
                # 🎯 整合架構：直接提交到ThreadPoolExecutor
                work_queue_size_before = self.thread_pool._work_queue.qsize() if hasattr(self.thread_pool, '_work_queue') else 0
                
                # 直接提交任務到ThreadPoolExecutor
                future = self.thread_pool.submit(self._process_frame_task, frame)
                
                work_queue_size_after = self.thread_pool._work_queue.qsize() if hasattr(self.thread_pool, '_work_queue') else 0
                success_count += 1

                logger.info(f"� [PIPELINE] 提交幀 #{frame_id} 到ThreadPool，work_queue: {work_queue_size_before}→{work_queue_size_after}")

                # 智能流控：如果ThreadPoolExecutor的work_queue太多任務，暫停一下
                if work_queue_size_after > 50:  # 閾值設為50個待處理任務
                    time.sleep(0.02)  # 20ms微暫停
                    
            except Exception as e:
                frame_id = frame.get('frame_id', i) if isinstance(frame, dict) else i
                logger.error(f"❌ [DIRECT_SUBMIT] 提交幀 #{frame_id} 失敗: {e}")
                break
                
        return success_count
    
    def _consumer_loop(self):
        """Video模式Consumer - 完整顯示"""
        consumed_count = 0
        while self.running:
            try:
                # 添加詳細的get操作日誌
                queue_size_before = self.output_queue.qsize()
                result = self.output_queue.get(timeout=1.0)
                queue_size_after = self.output_queue.qsize()
                logger.info(f"📤 [OUTPUT_QUEUE_GET] 取得結果 ({queue_size_before}→{queue_size_after}/{self.output_queue.maxsize}), 提交第 {consumed_count + 1} 個結果到Consumer")

                if result is None:
                    logger.info("⚠️ Consumer收到終止信號 (None result)")
                    break

                # 調用consumer的consume方法處理結果
                try:
                    self.consumer.consume(result)
                except Exception as e:
                    logger.error(f"❌ [CONSUMER_PROCESS] Consumer處理第 {consumed_count + 1} 個結果時出錯: {e}")
                    
                self.pipeline_frame_counter += 1
                consumed_count += 1
                
                if consumed_count % 20 == 0:  # 每20個結果記錄一次
                    output_size = self.output_queue.qsize()
                    logger.info(f"🖥️ Video Consumer狀態: 已消費 {consumed_count} 個結果，Output Queue: {output_size}")
                
                # 更新Consumer狀態 - 使用Consumer的實際顯示幀數
                consumer_frame_count = getattr(self.consumer.stats, 'total_displayed', consumed_count) if hasattr(self.consumer, 'stats') and self.consumer.stats else consumed_count
                self.timeline_debugger.update_consumer_state(
                    active=True, 
                    frame_count=consumer_frame_count
                )
                
            except Exception as e:
                if self.producer_finished and self.output_queue.empty():
                    logger.info("🏁 Producer已完成且Output Queue為空，Consumer準備結束")
                    break
                logger.debug(f"⚠️ Consumer超時等待: {e}")

        logger.info(f"✅ VIDEO CONSUMER 完成: 總共消費 {consumed_count} 個結果")

    def _handle_completion(self):
        """Video模式完成處理 - 整合架構等待完整"""
        # 🎯 整合架構：等待ThreadPoolExecutor和OUTPUT_QUEUE清空
        wait_start_time = time.time()
        while not self.output_queue.empty():
            # 檢查ThreadPoolExecutor是否有待處理任務
            thread_pool_queue_size = 0
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                if hasattr(self.thread_pool, '_work_queue'):
                    thread_pool_queue_size = self.thread_pool._work_queue.qsize()
                    
            output_size = self.output_queue.qsize()
            logger.debug(f"⏳ 等待隊列清空... ThreadPool work_queue: {thread_pool_queue_size}, Output: {output_size}")
            time.sleep(0.1)
            
            # 避免無限等待
            if time.time() - wait_start_time > 30:  # 30秒超時
                logger.warning("⚠️ 等待隊列清空超時，強制結束")
                break
                
            # 如果ThreadPool和OUTPUT_QUEUE都空了就結束
            if thread_pool_queue_size == 0 and output_size == 0:
                break
        
        # 停止WorkerPool並等待完成
        self.worker_pool.stop()
        self.output_queue.put(None)
    
    def _adjust_parameters_if_needed(self, metrics):
        """Video模式參數調整"""
        if len(self.performance_history) < 5:
            return
        
        recent_metrics = self.performance_history[-5:]
        avg_thread_pool_util = sum(m['thread_pool_util'] for m in recent_metrics) / len(recent_metrics)
        
        high_watermark = self.adaptive_params["queue_high_watermark"] / 100.0
        low_watermark = self.adaptive_params["queue_low_watermark"] / 100.0
        
        if avg_input_util > high_watermark:
            # 減少batch size來避免隊列過滿
            self.preload_batch_size = max(1, self.preload_batch_size - 1)
            logger.debug(f"Video mode: Reduced batch size to {self.preload_batch_size}")
        elif avg_input_util < low_watermark:
            # 增加batch size來提高throughput
            max_batch = self.adaptive_params["max_queue_size"] // 10
            self.preload_batch_size = min(max_batch, self.preload_batch_size + 1)
            logger.debug(f"Video mode: Increased batch size to {self.preload_batch_size}")


class CameraPipeline(BasePipeline):
    """
    Camera模式Pipeline - 實時性優先
    
    特點：
    - 協商式流控
    - 背壓檢測和智能丟幀
    - 快速停止機制
    """
    
    def _mode_specific_init(self):
        """Camera模式特定初始化"""
        logger.info("📸 ===== CAMERA模式 特定初始化 =====")
        
        # 流控制和能力協商參數
        self.flow_control_enabled = True
        self.components_capacity = {}
        self.last_capacity_negotiation_time = time.time()
        self.capacity_negotiation_interval = 2.0
        
        # 背壓檢測參數
        self.backpressure_threshold = 0.8  # 80%隊列利用率觸發背壓
        self.frame_drop_enabled = True
        self.consecutive_drops = 0
        self.max_consecutive_drops = 5
        
        logger.info(f"🔧 Camera模式參數配置:")
        logger.info(f"  └─ 流控制: {'啟用' if self.flow_control_enabled else '停用'}")
        logger.info(f"  └─ 背壓閾值: {self.backpressure_threshold * 100}%")
        logger.info(f"  └─ 智能丟幀: {'啟用' if self.frame_drop_enabled else '停用'}")
        logger.info(f"  └─ 最大連續丟幀: {self.max_consecutive_drops}")
        logger.info(f"  └─ 能力協商間隔: {self.capacity_negotiation_interval}秒")
        logger.info("✅ Pipeline初始化完成")
    
    def _producer_loop(self):
        """Camera模式Producer - 協商式流控"""
        logger.info("📸 ===== CAMERA PRODUCER 啟動 =====")
        logger.info("📝 Camera Producer策略: 協商式流控，實時性優先，智能丟幀")
        
        frame_count = 0
        dropped_frames = 0
        
        self.timeline_debugger.update_producer_state(active=True, frame_count=0)
        logger.info("📈 Timeline狀態更新: Camera Producer已啟動")
        
        try:
            logger.info("🎬 開始即時幀捕獲...")
            for frame in self.producer:
                if not self.running:
                    logger.warning("⚠️ Camera Producer收到停止信號")
                    break
                
                # 檢查背壓並決定是否丟幀
                should_drop = self._should_drop_frame()
                if should_drop:
                    dropped_frames += 1
                    self.consecutive_drops += 1
                    logger.debug(f"🗑️ Camera模式智能丟幀: 第 {dropped_frames} 幀 (連續丟幀: {self.consecutive_drops})")
                    continue
                else:
                    self.consecutive_drops = 0
                
                # 🎯 整合架構：Camera模式直接提交到ThreadPoolExecutor
                try:
                    work_queue_size_before = self.thread_pool._work_queue.qsize() if hasattr(self.thread_pool, '_work_queue') else 0
                    logger.info(f"� [CAMERA_DIRECT_SUBMIT] Camera準備直接提交幀 #{frame_count + 1}，ThreadPool work_queue: {work_queue_size_before}")
                    
                    # 直接提交任務到ThreadPoolExecutor
                    future = self.thread_pool.submit(self._process_frame_task, frame)
                    
                    work_queue_size_after = self.thread_pool._work_queue.qsize() if hasattr(self.thread_pool, '_work_queue') else 0
                    frame_count += 1
                    logger.info(f"✅ [CAMERA_DIRECT_SUBMIT] Camera成功提交幀 work_queue: {work_queue_size_before}→{work_queue_size_after}")
                except Exception as e:
                    # 提交失敗時丟幀
                    dropped_frames += 1
                    logger.warning(f"🗑️ [CAMERA_DIRECT_SUBMIT] Camera提交失敗丟幀: 第 {dropped_frames} 幀 - {e}")
                
                if (frame_count + dropped_frames) % 100 == 0:  # 每100幀記錄一次
                    drop_rate = dropped_frames / (frame_count + dropped_frames) * 100
                    work_queue_size = self.thread_pool._work_queue.qsize() if hasattr(self.thread_pool, '_work_queue') else 0
                    logger.info(f"📊 Camera Producer狀態: 處理 {frame_count} 幀, 丟幀 {dropped_frames} ({drop_rate:.1f}%), ThreadPool work_queue: {work_queue_size}")
                
                # 更新Producer狀態
                self.producer_last_activity = time.time()
                self.timeline_debugger.update_producer_state(active=True, frame_count=frame_count)
                
                # 能力協商
                self._negotiate_capacity_if_needed()
                
        finally:
            self.producer_finished = True
            self.timeline_debugger.update_producer_state(active=False, frame_count=frame_count)
            total_frames = frame_count + dropped_frames
            drop_rate = (dropped_frames / total_frames * 100) if total_frames > 0 else 0
            logger.info(f"✅ CAMERA PRODUCER 完成: 處理 {frame_count} 幀, 丟幀 {dropped_frames} ({drop_rate:.1f}%)")
            logger.info("🎯 Camera模式優先保證實時性能")
    
    def _should_drop_frame(self):
        """檢查是否應該丟幀（背壓檢測）"""
        if not self.frame_drop_enabled:
            return False
        
        # 🎯 整合架構：檢查ThreadPoolExecutor和OUTPUT_QUEUE壓力
        thread_pool_util = 0.0
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            if hasattr(self.thread_pool, '_work_queue') and hasattr(self.thread_pool, '_max_workers'):
                work_queue_size = self.thread_pool._work_queue.qsize()
                max_workers = self.thread_pool._max_workers
                # 以5倍worker數量作為滿載基準
                thread_pool_util = work_queue_size / (max_workers * 5.0)
                
        output_util = self.output_queue.qsize() / self.output_queue.maxsize
        
        # 背壓條件
        if thread_pool_util > self.backpressure_threshold or output_util > self.backpressure_threshold:
            logger.debug(f"🔴 背壓檢測觸發: ThreadPool={thread_pool_util:.2f}, Output={output_util:.2f}, 閾值={self.backpressure_threshold:.2f}")
            return True
        
        # 連續丟幀限制
        if self.consecutive_drops >= self.max_consecutive_drops:
            logger.debug(f"⚠️ 達到最大連續丟幀限制: {self.max_consecutive_drops}")
            return False
        
        return False
    
    def _consumer_loop(self):
        """Camera模式Consumer - 智能丟幀"""
        logger.info("🖥️ ===== CAMERA CONSUMER 啟動 =====")
        logger.info("📝 Camera Consumer策略: 智能顯示頻率控制，30 FPS目標")
        
        last_display_time = time.time()
        target_display_interval = 1.0 / 30  # 30 FPS目標
        consumed_count = 0
        skipped_count = 0
        
        logger.info(f"🎯 目標顯示間隔: {target_display_interval:.3f}秒 (30 FPS)")
        
        while self.running:
            try:
                # 添加詳細的get操作日誌
                queue_size_before = self.output_queue.qsize()
                logger.info(f"📤 [OUTPUT_QUEUE_GET] Camera準備取得結果，當前大小: {queue_size_before}")
                
                result = self.output_queue.get(timeout=0.5)  # 更短超時
                
                queue_size_after = self.output_queue.qsize()
                logger.info(f"📤 [OUTPUT_QUEUE_GET] Camera成功取得結果 {queue_size_before}→{queue_size_after}")
                
                if result is None:
                    logger.info("⚠️ Camera Consumer收到終止信號")
                    break
                
                current_time = time.time()
                time_since_last = current_time - last_display_time
                
                # 智能顯示頻率控制
                if time_since_last >= target_display_interval:
                    logger.info(f"�️ [CONSUMER_PROCESS] Camera Consumer顯示第 {consumed_count + 1} 個結果")
                    self.consumer.consume(result)
                    logger.info(f"✅ [CONSUMER_PROCESS] Camera Consumer成功顯示結果")
                    last_display_time = current_time
                    self.pipeline_frame_counter += 1
                    consumed_count += 1
                    
                    # 更新Consumer狀態 - 使用Consumer的實際顯示幀數
                    consumer_frame_count = getattr(self.consumer.stats, 'total_displayed', consumed_count) if hasattr(self.consumer, 'stats') and self.consumer.stats else consumed_count
                    self.timeline_debugger.update_consumer_state(
                        active=True, 
                        frame_count=consumer_frame_count
                    )
                else:
                    # 跳過此幀以維持實時性
                    skipped_count += 1
                    logger.debug(f"⏭️ 跳過結果維持實時性 (時間差: {time_since_last:.3f}s)")
                
                if (consumed_count + skipped_count) % 50 == 0:
                    output_size = self.output_queue.qsize()
                    skip_rate = skipped_count / (consumed_count + skipped_count) * 100
                    logger.info(f"🖥️ Camera Consumer狀態: 顯示 {consumed_count}, 跳過 {skipped_count} ({skip_rate:.1f}%), Queue: {output_size}")
                
            except Exception as e:
                logger.info(f"📤 [OUTPUT_QUEUE_GET] Camera Queue為空，等待超時")
                logger.debug(f"⚠️ Camera Consumer短超時: {e}")
                if self.producer_finished and self.output_queue.empty():
                    break
        
        total_results = consumed_count + skipped_count
        skip_rate = (skipped_count / total_results * 100) if total_results > 0 else 0
        logger.info(f"✅ CAMERA CONSUMER 完成: 顯示 {consumed_count}, 跳過 {skipped_count} ({skip_rate:.1f}%)")
        logger.info("🎯 Camera模式保持30 FPS實時顯示")
    
    def _handle_completion(self):
        """Camera模式完成處理 - 快速停止"""
        logger.info("🏁 ===== CAMERA模式 完成處理 =====")
        logger.info("⚡ Camera模式策略: 快速停止，不等待所有幀完成")
        
        # 🎯 整合架構：Camera模式快速停止，檢查ThreadPoolExecutor和OUTPUT_QUEUE狀態
        thread_pool_remaining = 0
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            if hasattr(self.thread_pool, '_work_queue'):
                thread_pool_remaining = self.thread_pool._work_queue.qsize()
                
        output_remaining = self.output_queue.qsize()
        logger.info(f"📊 停止時隊列狀態: ThreadPool work_queue: {thread_pool_remaining}, Output Queue: {output_remaining}")
        
        logger.info("🛑 快速停止WorkerPool...")
        self.worker_pool.stop()
        logger.info("📤 發送Consumer終止信號...")
        self.output_queue.put(None)
        logger.info("✅ Camera模式快速停止完成 - 優先實時性能")
    
    def _negotiate_capacity_if_needed(self):
        """能力協商機制"""
        current_time = time.time()
        if current_time - self.last_capacity_negotiation_time < self.capacity_negotiation_interval:
            return
        
        # 收集組件能力信息
        metrics = self._collect_recent_performance_metrics()
        
        # 更新能力報告
        self.components_capacity = {
            'producer_fps': metrics['current_fps'],
            'worker_utilization': metrics['worker_utilization'],
            'queue_pressure': max(metrics['thread_pool_util'], metrics['output_queue_util'])
        }
        
        logger.debug(f"🤝 能力協商更新: FPS={self.components_capacity['producer_fps']:.1f}, "
                   f"Worker利用率={self.components_capacity['worker_utilization']:.2f}, "
                   f"Queue壓力={self.components_capacity['queue_pressure']:.2f}")
        
        # 根據能力調整參數
        if self.components_capacity['queue_pressure'] > 0.8:
            if not self.frame_drop_enabled:
                self.frame_drop_enabled = True
                logger.info("🔴 啟用智能丟幀 - 檢測到高Queue壓力")
        elif self.components_capacity['queue_pressure'] < 0.3:
            if self.frame_drop_enabled:
                self.frame_drop_enabled = False
                logger.info("🟢 停用智能丟幀 - Queue壓力正常")
        
        self.last_capacity_negotiation_time = current_time


def create_pipeline(producer, worker_pool, consumer):
    """
    Pipeline工廠函數
    
    根據producer.mode自動選擇適當的Pipeline類型
    """
    mode = getattr(producer, 'mode', 'camera')

    if mode == 'video':
        logger.info("� ============================================================")
        logger.info("🔄 Video Pipeline初始化開始")
        logger.info("🔄 ============================================================")
        pipeline = VideoPipeline(producer, worker_pool, consumer)
    else:
        logger.info("� ============================================================")
        logger.info("🔄 Camera Pipeline初始化開始")
        logger.info("🔄 ============================================================")
        pipeline = CameraPipeline(producer, worker_pool, consumer)

    return pipeline
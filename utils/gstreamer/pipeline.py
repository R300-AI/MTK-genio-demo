"""
================================================================================
🏗        # 從WorkerPool獲取硬體資訊（避免重複檢測）
        logger.info("📋 步驟 2/4: 🔧 從WorkerPool取得硬體資訊與適應性參數...")
        if hasattr(self.worker_pool, '_performance_level'):
            hardware_tier = getattr(self.worker_pool, '_performance_level', 'MEDIUM')
            cpu_cores = getattr(self.worker_pool, '_cpu_cores', 4)
            memory_gb = getattr(self.worker_pool, '_memory_gb', 8.0)
            logger.info(f"📊 硬體資訊來源: WorkerPool")
            logger.info(f"📊 硬體性能等級: {hardware_tier} (CPU核心: {cpu_cores}, 記憶體: {memory_gb:.1f}GB)")
        else:
            # 如果 WorkerPool 沒有硬體檢測，使用預設值
            hardware_tier = 'MEDIUM'
            logger.warning("⚠️ WorkerPool未提供硬體資訊，使用預設等級: MEDIUM")
        
        # 根據硬體等級生成Pipeline專用的適應性參數
        self.adaptive_params = self._generate_pipeline_adaptive_params(hardware_tier)
        logger.info(f"📊 Pipeline適應性參數: {self.adaptive_params}")line 架構設計
================================================================================

Pipeline類採用繼承架構，支援Video模式（完整性優先）和Camera模式（實時性優先）。

🎯 核心組件：
┌─────────────┬──────────────────┬─────────────────────────────────────────┐
│ 📸 Producer │ 幀生產與推送     │ Video:確保無丟幀 / Camera:協商式流控    │
│ ⚙️ Worker   │ 幀處理與推理     │ Video:硬體適應性 / Camera:背壓檢測      │
│ 🖥️ Consumer │ 結果顯示與輸出   │ Video:完整保證 / Camera:實時優化        │
└─────────────┴──────────────────┴─────────────────────────────────────────┘

📊 資料流向：Producer ──[input_queue]──> Worker ──[output_queue]──> Consumer

🎯 繼承關係：
                    BasePipeline (抽象基類)
                    ├── 通用初始化和硬體檢測
                    ├── 統一執行流程模板
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
from queue import Queue
from abc import ABC, abstractmethod
from .metric import TimelineLogger, HardwarePerformanceLogger

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gstreamer_demo.log', mode='w', encoding='utf-8')
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
        
        # 硬體性能檢測和適應性參數
        logger.info("� 步驟 2/4: �🔧 初始化硬體檢測器與適應性參數...")
        self.hardware_detector = HardwarePerformanceLogger()
        self.adaptive_params = self.hardware_detector.get_adaptive_parameters()
        logger.info(f"📊 適應性參數: {self.adaptive_params}")
        
        # Timeline調試工具
        logger.info("� 步驟 3/4: ⚙️ 初始化Timeline調試工具與性能監控...")
        self.timeline_debugger = TimelineLogger()
        
        # Queue配置（使用適應性參數）
        max_queue_size = self.adaptive_params["max_queue_size"]
        logger.info(f"🔧 配置Queue - 最大大小: {max_queue_size}")
        self.input_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        self.running = False
        logger.info("✅ Input/Output Queue 創建完成")
        
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
        """根據硬體等級生成Pipeline專用的適應性參數"""
        params_map = {
            "EXTREME": {
                "max_queue_size": 100,
                "fps_check_interval": 50,
                "batch_timeout": 0.001,
                "queue_high_watermark": 80,
                "queue_low_watermark": 60
            },
            "HIGH": {
                "max_queue_size": 80,
                "fps_check_interval": 35,
                "batch_timeout": 0.002,
                "queue_high_watermark": 70,
                "queue_low_watermark": 50
            },
            "MEDIUM": {
                "max_queue_size": 60,
                "fps_check_interval": 25,
                "batch_timeout": 0.005,
                "queue_high_watermark": 60,
                "queue_low_watermark": 40
            },
            "LOW": {
                "max_queue_size": 40,
                "fps_check_interval": 15,
                "batch_timeout": 0.01,
                "queue_high_watermark": 50,
                "queue_low_watermark": 30
            }
        }
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
        """更新時間軸狀態信息"""
        try:
            # 更新Queue狀態
            self.timeline_debugger.update_queue_states(
                input_size=self.input_queue.qsize(),
                output_size=self.output_queue.qsize()
            )
            
            # 更新Worker狀態
            if hasattr(self.worker_pool, 'workers'):
                for i, worker in enumerate(getattr(self.worker_pool, 'workers', [])):
                    worker_id = f"W{i+1}"
                    is_active = getattr(worker, 'is_busy', False) or getattr(worker, 'active', False)
                    task_count = getattr(worker, 'task_count', 0)
                    self.timeline_debugger.update_worker_state(worker_id, active=is_active, task_count=task_count)
            elif hasattr(self.worker_pool, 'active_workers') and hasattr(self.worker_pool, 'num_workers'):
                active_workers = getattr(self.worker_pool, 'active_workers', 0)
                total_workers = getattr(self.worker_pool, 'num_workers', 1)
                
                for i in range(total_workers):
                    worker_id = f"W{i+1}"
                    is_active = i < active_workers
                    self.timeline_debugger.update_worker_state(worker_id, active=is_active)
            
        except Exception as e:
            logger.debug(f"Timeline state update error: {e}")
    
    def run(self):
        """Template Method - 統一執行流程模板"""
        self.running = True
        logger.info(" ---------------------------------------------------------")
        logger.info("🚀 開始執行")
        self._start_timeline_logging()
        
        def result_handler(result):
            if result is not None:
                try:
                    self.output_queue.put(result, timeout=1.0)
                    self.timeline_debugger.update_consumer_state(active=True)
                    logger.info(f"📤 WorkerPool結果已加入output_queue (當前大小: {self.output_queue.qsize()})")
                except Exception as e:
                    logger.error(f"❌ PIPELINE_CALLBACK: Failed to queue result: {e}")
            else:
                logger.warning("⚠️ WorkerPool返回了空結果 (None)")

        logger.info("🔧 正在啟動WorkerPool...")
        self.worker_pool.start(result_handler)
        logger.info("🔧 正在啟動Consumer顯示...")
        self.consumer.start_display()

        producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        worker_thread.start()
        producer_thread.start() 
        consumer_thread.start()
        
        try:
            logger.info("⏳ 等待Producer執行緒完成...")
            producer_thread.join()
            logger.info("✅ Producer執行緒已完成")
            
            # 模式特定的完成處理
            logger.info(f"🏁 執行{self.__class__.__name__}完成處理...")
            self._handle_completion()
            
            logger.info("⏳ 等待Consumer執行緒完成...")
            consumer_thread.join()
            logger.info("✅ Consumer執行緒已完成")
            
            logger.info("🛑 停止Consumer顯示...")
            self.consumer.stop_display()
            
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
            
        except Exception as e:
            logger.error(f"❌ PIPELINE_RUN: Error during execution: {e}")
        finally:
            logger.info("🧹 執行清理程序...")
            self.running = False
            self.worker_pool.stop()
            if hasattr(self, 'timeline_thread'):
                self.timeline_thread.join(timeout=1.0)
            logger.info("✅ Pipeline執行完成!")
            logger.info("📊 ============================================================")
    
    @abstractmethod
    def _producer_loop(self):
        """Producer邏輯（子類實現）"""
        pass
    
    @abstractmethod
    def _worker_loop(self):
        """Worker邏輯（子類實現）"""
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
        """收集最近的性能指標用於適應性調整"""
        current_time = time.time()
        
        # 收集隊列狀態
        input_queue_utilization = self.input_queue.qsize() / self.input_queue.maxsize
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
        
        # 記錄性能數據
        performance_metrics = {
            'timestamp': current_time,
            'input_queue_util': input_queue_utilization,
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
        """Video模式Producer - 確保無丟幀"""
        logger.info("🎬 ===== VIDEO PRODUCER 啟動 =====")
        logger.info("📝 Video Producer策略: 完整性優先，確保無丟幀，順序處理")
        
        frame_count = 0
        last_adjustment_time = time.time()
        batch_timeout = self.adaptive_params["batch_timeout"]
        # 預載相關變數
        frame_buffer = []
        max_buffer_size = self.preload_batch_size * 2
        self.timeline_debugger.update_producer_state(active=True, frame_count=0)
        
        try:
            logger.info("🎬 開始影片幀讀取循環...")
            for frame in self.producer:
                if not self.running:
                    logger.warning("⚠️ Producer收到停止信號，中斷幀讀取")
                    break
                
                frame_buffer.append(frame)
                frame_count += 1
                
                # 添加詳細的幀處理日誌
                if frame_count <= 5:
                    logger.info(f"📦 第 {frame_count} 幀已加入緩衝區，緩衝區大小: {len(frame_buffer)}")

                # 批次處理並預載
                if len(frame_buffer) >= self.preload_batch_size:
                    logger.debug(f"📦 執行批次處理: {len(frame_buffer)} 幀")
                    self._process_frame_batch_video(frame_buffer, batch_timeout)
                    frame_buffer = []
                
                # 更新Producer狀態
                self.producer_last_activity = time.time()
                self.timeline_debugger.update_producer_state(active=True, frame_count=frame_count)
                
                # 定期性能調整
                current_time = time.time()
                if current_time - last_adjustment_time > self.pipeline_fps_check_interval:
                    metrics = self._collect_recent_performance_metrics()
                    logger.debug(f"📊 性能指標更新: FPS={metrics['current_fps']:.2f}, "
                               f"Input Queue={metrics['input_queue_util']:.2f}, "
                               f"Worker利用率={metrics['worker_utilization']:.2f}")
                    self._adjust_parameters_if_needed(metrics)
                    last_adjustment_time = current_time
            
            # 處理剩餘frames
            if frame_buffer:
                logger.info(f"📦 處理剩餘的 {len(frame_buffer)} 幀")
                self._process_frame_batch_video(frame_buffer, batch_timeout)
            
        except StopIteration:
            logger.info("🏁 Producer迭代完成 (StopIteration)")
        except Exception as e:
            logger.error(f"❌ Producer循環發生錯誤: {e}")
        finally:
            self.producer_finished = True
            self.timeline_debugger.update_producer_state(active=False, frame_count=frame_count)
            logger.info(f"✅ VIDEO PRODUCER 完成: 總共處理 {frame_count} 幀")
    
    def _process_frame_batch_video(self, frame_batch, timeout):
        """處理Video模式的frame批次"""
        for frame in frame_batch:
            try:
                self.input_queue.put(frame, timeout=timeout)
            except Exception as e:
                logger.warning(f"Video mode frame put timeout: {e}")
                # Video模式重試機制
                time.sleep(0.01)
                try:
                    self.input_queue.put(frame, timeout=timeout * 2)
                except:
                    logger.error("Video mode frame lost despite retry")
    
    def _worker_loop(self):
        """Video模式Worker - 硬體適應性"""
        logger.info("⚙️ ===== VIDEO WORKER 啟動 =====")
        logger.info("📝 Video Worker策略: 硬體適應性，完整處理，順序保證")
        
        processed_count = 0
        
        while self.running or not self.input_queue.empty():
            try:
                frame = self.input_queue.get(timeout=1.0)
                if frame is None:
                    logger.info("⚠️ Worker收到終止信號 (None frame)")
                    break
                
                # 提交給WorkerPool處理
                logger.debug(f"📤 提交第 {processed_count + 1} 幀給WorkerPool處理")
                if processed_count < 5:  # 只記錄前5幀
                    logger.info(f"📤 提交第 {processed_count + 1} 幀給WorkerPool處理")
                
                self.worker_pool.submit(frame)
                processed_count += 1
                
                if processed_count % 20 == 0:  # 每20幀記錄一次
                    input_size = self.input_queue.qsize()
                    logger.info(f"⚙️ Video Worker狀態: 已處理 {processed_count} 幀，Input Queue: {input_size}")

            except Exception as e:
                if self.producer_finished and self.input_queue.empty():
                    logger.info("🏁 Producer已完成且Queue為空，Worker準備結束")
                    break
                logger.debug(f"⚠️ Worker超時等待: {e}")
        
        logger.info(f"✅ VIDEO WORKER 完成: 總共處理 {processed_count} 幀")
    
    def _consumer_loop(self):
        """Video模式Consumer - 完整顯示"""
        logger.info("🖥️ ===== VIDEO CONSUMER 啟動 =====")
        logger.info("📝 Video Consumer策略: 完整顯示，順序處理，無丟幀")
        
        consumed_count = 0
        while self.running:
            try:
                result = self.output_queue.get(timeout=1.0)
                if result is None:
                    logger.info("⚠️ Consumer收到終止信號 (None result)")
                    break
                
                logger.debug(f"📥 Consumer收到第 {consumed_count + 1} 個結果")
                if consumed_count < 5:  # 只記錄前5個結果
                    logger.info(f"📥 Consumer收到第 {consumed_count + 1} 個結果")
                
                # 調用consumer的consume方法處理結果
                try:
                    self.consumer.consume(result)
                    if consumed_count < 5:
                        logger.info(f"✅ Consumer成功處理第 {consumed_count + 1} 個結果")
                except Exception as e:
                    logger.error(f"❌ Consumer處理第 {consumed_count + 1} 個結果時出錯: {e}")
                    
                self.pipeline_frame_counter += 1
                consumed_count += 1
                
                if consumed_count % 20 == 0:  # 每20個結果記錄一次
                    output_size = self.output_queue.qsize()
                    logger.info(f"🖥️ Video Consumer狀態: 已消費 {consumed_count} 個結果，Output Queue: {output_size}")
                
                # 更新Consumer狀態
                self.timeline_debugger.update_consumer_state(
                    active=True, 
                    frame_count=self.pipeline_frame_counter
                )
                
            except Exception as e:
                if self.producer_finished and self.output_queue.empty():
                    logger.info("🏁 Producer已完成且Output Queue為空，Consumer準備結束")
                    break
                logger.debug(f"⚠️ Consumer超時等待: {e}")

        logger.info(f"✅ VIDEO CONSUMER 完成: 總共消費 {consumed_count} 個結果")

    def _handle_completion(self):
        """Video模式完成處理 - 等待完整"""
        # 等待所有frames被處理完畢
        wait_start_time = time.time()
        while not self.input_queue.empty() or not self.output_queue.empty():
            input_size = self.input_queue.qsize()
            output_size = self.output_queue.qsize()
            logger.debug(f"⏳ 等待隊列清空... Input: {input_size}, Output: {output_size}")
            time.sleep(0.1)
            
            # 避免無限等待
            if time.time() - wait_start_time > 30:  # 30秒超時
                logger.warning("⚠️ 等待隊列清空超時，強制結束")
                break
        
        # 停止WorkerPool並等待完成
        self.worker_pool.stop()
        self.output_queue.put(None)
    
    def _adjust_parameters_if_needed(self, metrics):
        """Video模式參數調整"""
        if len(self.performance_history) < 5:
            return
        
        recent_metrics = self.performance_history[-5:]
        avg_input_util = sum(m['input_queue_util'] for m in recent_metrics) / len(recent_metrics)
        
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
                
                # 嘗試非阻塞put
                try:
                    self.input_queue.put_nowait(frame)
                    frame_count += 1
                    logger.debug(f"📤 第 {frame_count} 幀已加入處理隊列")
                except:
                    # 隊列滿時丟幀
                    dropped_frames += 1
                    logger.debug(f"🗑️ Queue滿載丟幀: 第 {dropped_frames} 幀")
                
                if (frame_count + dropped_frames) % 100 == 0:  # 每100幀記錄一次
                    drop_rate = dropped_frames / (frame_count + dropped_frames) * 100
                    input_size = self.input_queue.qsize()
                    logger.info(f"📊 Camera Producer狀態: 處理 {frame_count} 幀, 丟幀 {dropped_frames} ({drop_rate:.1f}%), Queue: {input_size}")
                
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
        
        # 檢查隊列壓力
        input_util = self.input_queue.qsize() / self.input_queue.maxsize
        output_util = self.output_queue.qsize() / self.output_queue.maxsize
        
        # 背壓條件
        if input_util > self.backpressure_threshold or output_util > self.backpressure_threshold:
            logger.debug(f"🔴 背壓檢測觸發: Input={input_util:.2f}, Output={output_util:.2f}, 閾值={self.backpressure_threshold:.2f}")
            return True
        
        # 連續丟幀限制
        if self.consecutive_drops >= self.max_consecutive_drops:
            logger.debug(f"⚠️ 達到最大連續丟幀限制: {self.max_consecutive_drops}")
            return False
        
        return False
    
    def _worker_loop(self):
        """Camera模式Worker - 背壓檢測"""
        logger.info("⚙️ ===== CAMERA WORKER 啟動 =====")
        logger.info("📝 Camera Worker策略: 背壓檢測，快速響應")
        
        processed_count = 0
        
        while self.running or not self.input_queue.empty():
            try:
                frame = self.input_queue.get(timeout=0.5)  # 更短超時
                if frame is None:
                    logger.info("⚠️ Camera Worker收到終止信號")
                    break
                
                # 提交給WorkerPool處理
                logger.debug(f"📤 Camera Worker處理第 {processed_count + 1} 幀")
                self.worker_pool.submit(frame)
                processed_count += 1
                
                if processed_count % 30 == 0:  # Camera模式更頻繁記錄
                    input_size = self.input_queue.qsize()
                    logger.info(f"⚙️ Camera Worker狀態: 已處理 {processed_count} 幀，Input Queue: {input_size}")
                
            except Exception as e:
                if self.producer_finished and self.input_queue.empty():
                    break
                logger.debug(f"⚠️ Camera Worker短超時: {e}")
        
        logger.info(f"✅ CAMERA WORKER 完成: 總共處理 {processed_count} 幀")
    
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
                result = self.output_queue.get(timeout=0.5)  # 更短超時
                if result is None:
                    logger.info("⚠️ Camera Consumer收到終止信號")
                    break
                
                current_time = time.time()
                time_since_last = current_time - last_display_time
                
                # 智能顯示頻率控制
                if time_since_last >= target_display_interval:
                    logger.debug(f"📥 Camera Consumer顯示第 {consumed_count + 1} 個結果")
                    self.consumer.consume(result)
                    last_display_time = current_time
                    self.pipeline_frame_counter += 1
                    consumed_count += 1
                    
                    # 更新Consumer狀態
                    self.timeline_debugger.update_consumer_state(
                        active=True, 
                        frame_count=self.pipeline_frame_counter
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
        
        # Camera模式快速停止，不等待所有幀處理完成
        input_remaining = self.input_queue.qsize()
        output_remaining = self.output_queue.qsize()
        logger.info(f"📊 停止時隊列狀態: Input Queue: {input_remaining}, Output Queue: {output_remaining}")
        
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
            'queue_pressure': max(metrics['input_queue_util'], metrics['output_queue_util'])
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
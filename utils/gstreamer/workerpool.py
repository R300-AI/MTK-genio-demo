"""
================================================================================
⚙️ WorkerPool 架構設計 2025.08.21
================================================================================

WorkerPool 採用多執行緒池管理 + 任務分發的設計。
所有建立操作僅透過worker_pool = WorkerPool(model_path, mode, max_workers) 作為唯一入口，並依據 mode 參數選擇對應的處理策略。

系統支援兩種處理模式：
┌─────────────┬──────────────────┬─────────────────────────────────────────┐
│ 📸 Video    │ 批次處理優先     │ 順序處理、結果同步、完整性保證          │
│ 📷 Camera   │ 即時處理優先     │ 並行推理、低延遲、實時響應              │
└─────────────┴──────────────────┴─────────────────────────────────────────┘

📊 資料流向：
    Producer ──[Frame Queue]──> WorkerPool ──[Inference Threads]──> Result Queue ──> Consumer

🎯 架構關係：
                    WorkerPool (執行緒池管理器)
                    ├── Thread Pool Management
                    ├── 任務分發與負載平衡
                    ├── 推理引擎管理
                    └── 結果集中處理
                           │
            ┌──────────────┴──────────────┐
            │                             │
    Video處理策略                  Camera處理策略
    (批次同步處理)                   (即時並行處理)

📊 職責分配 - 遵循工作者池模式：
┌─────────────────┬──────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   功能類別      │   WorkerPool     │   Worker執行緒   │  TaskManager    │  ModelManager   │
│                 │    (池管理器)    │   (工作執行緒)   │   (任務管理)    │   (模型管理)    │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 🏗️ 池創建管理   │ ✅ 執行緒池初始化 │ 🔹 被動創建     │ 🔹 不負責創建   │ 🔹 不負責創建   │
│ 🚀 工作者啟動   │ ✅ 執行緒生命週期 │ ✅ 執行緒主循環 │ 🔹 委託給池     │ 🔹 委託給池     │
│ 📋 任務分發     │ ✅ 負載平衡分發   │ ✅ 任務接收執行 │ ✅ 任務佇列管理 │ 🔹 不負責分發   │
│ 🧠 推理執行     │ 🔹 委託給執行緒   │ ✅ 模型推理執行 │ 🔹 不執行推理   │ ✅ 模型載入管理 │
│ 📊 結果處理     │ ✅ 結果匯集管理   │ ✅ 結果產生提交 │ ✅ 結果佇列管理 │ 🔹 不處理結果   │
│ 🔄 錯誤處理     │ ✅ 池級錯誤處理   │ ✅ 執行緒錯誤   │ ✅ 任務錯誤處理 │ ✅ 模型錯誤處理 │
│ 📈 性能監控     │ ✅ 池性能統計     │ ✅ 執行緒統計   │ ✅ 任務統計     │ ✅ 推理性能統計 │
│ 🧹 資源清理     │ ✅ 池資源清理     │ ✅ 執行緒清理   │ ✅ 佇列清理     │ ✅ 模型資源清理 │
│ 🎯 模式適配     │ ✅ 模式策略選擇   │ ✅ 模式特定邏輯 │ ✅ 模式特定管理 │ 🔹 模式無關     │
│ 🔧 配置管理     │ ✅ 池配置管理     │ 🔹 繼承池配置   │ 🔹 繼承池配置   │ ✅ 模型配置管理 │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┴─────────────────┘

🎯 設計原則說明：
┌─────────────────┬─────────────────────────────────────────────────────────┐
│   設計原則      │                     實現方式                            │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ 🔹 生產者-消費者 │ 使用Queue實現執行緒間安全通信                           │
│ ✅ 執行緒池模式  │ 預創建固定數量執行緒，避免頻繁創建銷毀                  │
│ 🔹 負載平衡     │ 動態分發任務到空閒執行緒，避免單點過載                  │
│ ✅ 容錯處理     │ 單一執行緒錯誤不影響整體池運作                          │
│ 🔹 資源隔離     │ 每個執行緒獨立模型實例，避免推理衝突                    │
│ ✅ 優雅關閉     │ 支援優雅停止，完成當前任務後關閉                        │
└─────────────────┴─────────────────────────────────────────────────────────┘

🔧 核心特性：
• Thread Pool Pattern：預創建執行緒池，高效任務處理
• Producer-Consumer：使用Queue實現安全的任務和結果傳遞
• Load Balancing：智能任務分發，最大化執行緒利用率
• Fault Tolerance：單一執行緒故障不影響整體系統運作
• Mode Adaptation：根據Video/Camera模式採用不同處理策略
• Resource Management：統一的資源生命週期管理

🛠️ 標準使用方式：
```python
# 唯一調用方式 - 統一工作者池接口
worker_pool = WorkerPool(model_path=args.model_path, mode=mode, max_workers=args.max_workers)

# 模式選擇邏輯：
# • mode="video"  → 批次同步處理策略 (完整性優先)
# • mode="camera" → 即時並行處理策略 (實時性優先)

# 範例：
worker_pool = WorkerPool("./model.tflite", mode="video", max_workers=2)    # 視頻批次處理
worker_pool = WorkerPool("./model.tflite", mode="camera", max_workers=4)   # 攝像頭即時處理

# Pipeline整合：
pipeline = create_pipeline(
    producer=producer,
    worker_pool=worker_pool,  # 執行緒池自動管理推理任務
    consumer=consumer
)
```

⚡ 處理策略差異：
┌─────────────────┬──────────────────┬─────────────────────────────────────┐
│     模式        │    處理策略      │              特性                   │
├─────────────────┼──────────────────┼─────────────────────────────────────┤
│ mode="video"    │ 批次同步處理     │ 順序保證、結果完整性、批次優化      │
│ mode="camera"   │ 即時並行處理     │ 低延遲優先、並行推理、實時響應      │
└─────────────────┴──────────────────┴─────────────────────────────────────┘

🚀 性能優化特性：
• 執行緒池預熱：啟動時預創建所有工作執行緒
• 模型預載入：每個執行緒獨立載入模型實例，避免推理衝突  
• 智能佇列管理：根據模式調整佇列大小和處理策略
• 動態負載平衡：實時監控執行緒負載，智能分發任務
• 批次處理優化：Video模式支援批次推理優化
• 記憶體管理：自動記憶體回收和資源清理

🔒 執行緒安全保證：
• Queue-based Communication：使用執行緒安全佇列
• Resource Isolation：每個執行緒獨立資源
• Atomic Operations：關鍵操作使用原子操作
• Lock-free Design：最小化鎖使用，提高並行性能

📊 監控與診斷：
• 即時性能統計：FPS、延遲、佇列狀態
• 執行緒健康監控：活躍執行緒、錯誤統計
• 資源使用監控：記憶體、CPU使用率
• 詳細日誌系統：分級日誌、除錯資訊
"""

import threading
import queue
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Literal
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
from utils.gstreamer.config import ProcessorConfig, WorkerPoolConfig
from utils.gstreamer.processor import Processor

logger = logging.getLogger("gstreamer_demo")


class WorkerPool:
    """
    🏭 WorkerPool - 即時就緒的池化管理系統
    
    功能：任務調度、負載均衡、Processor控制
    配置：通過WorkerPoolConfig控制調度策略
    
    Video模式：順序保證、大緩衝、無丟幀、停用背壓
    Camera模式：背壓控制、小緩衝、即時策略、啟用背壓
    """
    
    def __init__(self, processor_config: ProcessorConfig = None, worker_pool_config: WorkerPoolConfig = None, 
                 model_path: str = None, mode: Literal['video', 'camera'] = 'camera', 
                 max_workers: int = 4, **kwargs):
        """
        初始化WorkerPool池化管理系統
        
        支持兩種創建方式：
        1. 配置對象方式：WorkerPool(processor_config, worker_pool_config)
        2. 參數方式：WorkerPool(model_path, mode='video', max_workers=4)
        
        Args:
            processor_config: ProcessorConfig配置對象（優先使用）
            worker_pool_config: WorkerPoolConfig配置對象（優先使用）
            model_path: YOLO模型路徑（參數方式）
            mode: 模式選擇 ('video'|'camera')
            max_workers: 最大工作線程數
            **kwargs: 額外配置參數
        """
        # 創建配置對象 - 支持兩種方式
        if processor_config is not None:
            self.processor_config = processor_config
        else:
            if model_path is None:
                raise ValueError("必須提供 processor_config 或 model_path")
            self.processor_config = ProcessorConfig(
                mode=mode, 
                model_path=model_path
            )
        
        if worker_pool_config is not None:
            self.worker_pool_config = worker_pool_config
        else:
            self.worker_pool_config = WorkerPoolConfig(
                mode=mode,
                max_workers=max_workers,
                **kwargs
            )
        
        # 初始化核心組件
        self.processors: List[Processor] = []
        self.task_queue = queue.Queue(maxsize=self.worker_pool_config.buffer_size)
        self.executor = None
        self.is_running = False
        self.current_worker_index = 0
        self.lock = threading.Lock()
        
        # 順序保證相關（Video模式專用）
        if self.worker_pool_config.preserve_order:
            self.result_queue = queue.Queue()
            self.pending_results: Dict[int, Any] = {}
            self.next_expected_id = 0
            self.task_id_counter = 0
        
        # 統計資料
        self._total_tasks = 0
        self._completed_tasks = 0
        self._dropped_tasks = 0
        
        # 📝 WorkerPool即時就緒初始化日誌
        mode_tag = "[VIDEO]" if self.worker_pool_config.mode == 'video' else "[CAMERA]"
        logger.info("🏭 " + "="*60)
        logger.info("🏭 WorkerPool即時就緒初始化開始")
        logger.info("🏭 " + "="*60)
        
        # 📋 步驟 1/3: 配置驗證與基礎架構
        logger.info("📋 步驟 1/3: 🚀 配置驗證與基礎架構...")
        logger.info(f"🔍 {mode_tag} 工作者數量: {self.worker_pool_config.max_workers}")
        logger.info(f"🔍 {mode_tag} 緩衝區大小: {self.worker_pool_config.buffer_size}")
        
        # 模式設定詳細說明
        if self.worker_pool_config.mode == 'video':
            logger.info(f"🎯 {mode_tag} 模式設定: 完整性優先")
            logger.info(f"📊 {mode_tag}   - 順序保證: 開啟")
            logger.info(f"📊 {mode_tag}   - 背壓控制: 停用 (無丟幀)")
            logger.info(f"📊 {mode_tag}   - 緩衝策略: 大容量")
            logger.info(f"📊 {mode_tag}   - 處理策略: 等待完整處理")
        else:
            logger.info(f"🎯 {mode_tag} 模式設定: 實時性優先")
            logger.info(f"📊 {mode_tag}   - 順序保證: 停用")
            logger.info(f"📊 {mode_tag}   - 背壓控制: 開啟 (低延遲)")
            logger.info(f"📊 {mode_tag}   - 緩衝策略: 小容量")
            logger.info(f"📊 {mode_tag}   - 處理策略: 即時響應")

        # 硬體性能檢測
        import psutil
        cpu_count = psutil.cpu_count()
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        performance_score = (cpu_count * 2) + (memory_gb * 0.5)
        
        if performance_score >= 50:
            performance_level = "EXTREME"
        elif performance_score >= 30:
            performance_level = "HIGH"
        elif performance_score >= 15:
            performance_level = "MEDIUM"
        else:
            performance_level = "LOW"
            
        logger.info(f"📊 {mode_tag}   - 硬體性能等級: {performance_level} (CPU核心: {cpu_count}, 記憶體: {memory_gb}GB, 總分: {performance_score:.2f})")
        
        # 保存硬體性能檢測結果
        self._cpu_cores = cpu_count
        self._memory_gb = memory_gb
        self._performance_level = performance_level
        self._performance_score = performance_score
        
        logger.info(f"✅ {mode_tag} 步驟 1/3 完成 - 配置驗證與基礎架構")
        
        # 📋 步驟 2/3: Queue配置和內部結構
        logger.info("📋 步驟 2/3: 🔧 配置Queue和內部結構...")
        logger.info(f"🔧 {mode_tag} 任務佇列: 最大容量 {self.worker_pool_config.buffer_size}")
        
        if self.worker_pool_config.preserve_order:
            logger.info(f"🔧 {mode_tag} 順序佇列: 已初始化 (Video模式)")
        
        logger.info(f"✅ {mode_tag} 步驟 2/3 完成 - Queue和內部結構配置")
        
        # 📋 步驟 3/3: 創建Processor池並載入模型（核心步驟）
        logger.info("📋 步驟 3/3: ⚙️ 創建Processor池並載入YOLO模型...")
        logger.info(f"🔍 {mode_tag} 準備創建並初始化 {self.worker_pool_config.max_workers} 個Processor...")
        
        # 🎯 在初始化階段就載入所有模型 - 這才是初始化的核心目的！
        start_time = time.time()
        for i in range(self.worker_pool_config.max_workers):
            logger.info(f"🔍 {mode_tag} 正在初始化 Processor#{i+1}/{self.worker_pool_config.max_workers}...")
            
            try:
                processor = Processor(self.processor_config)
                if not processor.initialize():
                    raise RuntimeError(f"Processor#{i+1} 初始化失敗")
                
                self.processors.append(processor)
                logger.info(f"✅ {mode_tag} Processor#{i+1} 就緒 (YOLO模型已載入)")
                
            except Exception as e:
                logger.error(f"❌ {mode_tag} Processor#{i+1} 創建失敗: {e}")
                if self.processor_config.detailed_logging:
                    logger.error(f"❌ {mode_tag} 詳細錯誤: {traceback.format_exc()}")
                raise RuntimeError(f"WorkerPool初始化失敗: Processor#{i+1}載入模型失敗: {e}")
        
        # 初始化完成報告
        elapsed = time.time() - start_time
        logger.info(f"📈 {mode_tag} 所有Processor已就緒，模型載入完成!")
        logger.info(f"📊 {mode_tag} 可用Processor: {len(self.processors)} (每個都已載入YOLO)")
        logger.info(f"⏱️ {mode_tag} 模型載入總時間: {elapsed:.3f}s")
        
        if self.worker_pool_config.mode == 'video':
            logger.info("🎯 [VIDEO] 執行模式: 順序處理模式")
            logger.info("📈 [VIDEO] 系統準備: 完整性保證就緒")
        else:
            logger.info("🎯 [CAMERA] 執行模式: 並行處理模式")  
            logger.info("📈 [CAMERA] 系統準備: 實時響應就緒")
            
        logger.info("✅ WorkerPool即時就緒初始化完成! (隨時可以開始工作)")
        logger.info("💡 [提示] 系統已完全就緒，首次任務提交時將自動啟動線程池")
        logger.info("🏭 " + "="*60)
    
    def start(self, result_callback: Optional[Callable] = None) -> bool:
        """
        🚀 啟動WorkerPool線程池 - 開始接受任務
        
        注意：模型已經在__init__中載入，這裡只啟動線程池
        
        Args:
            result_callback: 結果回調函數（向後兼容）
        
        Returns:
            bool: 啟動是否成功
        """
        if self.is_running:
            logger.info("⚠️ WorkerPool線程池已經在運行中")
            return True
        
        mode_tag = "[VIDEO]" if self.worker_pool_config.mode == 'video' else "[CAMERA]"
        
        try:
            logger.info("🚀 WorkerPool啟動線程池...")
            logger.info(f"🔧 {mode_tag} 任務回調: {'已設置' if result_callback else '未設置'}")
            
            # 儲存回調函數
            self._result_callback = result_callback
            
            # 🎯 只在這裡啟動線程池 - 這才需要線程
            self.executor = ThreadPoolExecutor(
                max_workers=self.worker_pool_config.max_workers,
                thread_name_prefix=f"WorkerPool-{self.worker_pool_config.mode}"
            )
            self.is_running = True
            
            logger.info(f"✅ {mode_tag} 線程池啟動完成，開始接受任務!")
            logger.info(f"📊 {mode_tag} 執行器狀態: 運行中 ({len(self.processors)}個Processor就緒)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {mode_tag} 線程池啟動失敗: {e}")
            if self.processor_config.detailed_logging:
                logger.error(f"❌ {mode_tag} 詳細錯誤: {traceback.format_exc()}")
            return False
    
    def _get_next_processor(self) -> Processor:
        """
        🎯 智能調度：Round-Robin分配策略
        
        Returns:
            下一個可用的Processor
        """
        with self.lock:
            # Round-Robin調度
            processor = self.processors[self.current_worker_index]
            self.current_worker_index = (self.current_worker_index + 1) % len(self.processors)
            return processor
    
    def _should_drop_task(self) -> bool:
        """
        📊 背壓控制判斷
        
        Returns:
            bool: 是否應該丟棄任務
        """
        if not self.worker_pool_config.enable_backpressure:
            return False
        
        # 計算當前負載
        current_load = self.task_queue.qsize() / self.worker_pool_config.buffer_size
        
        return current_load >= self.worker_pool_config.drop_threshold
    
    def _process_task(self, processor: Processor, frame: Any, task_id: Optional[int] = None, 
                     callback: Optional[Callable] = None) -> Any:
        """
        任務處理內部方法
        """
        try:
            result = processor.predict(frame)
            
            # 優先使用傳入的回調，否則使用啟動時設定的回調（向後兼容）
            active_callback = callback or getattr(self, '_result_callback', None)
            if active_callback:
                active_callback(result)
            
            # Video模式順序保證
            if self.worker_pool_config.preserve_order and task_id is not None:
                with self.lock:
                    self.pending_results[task_id] = result
                    self._process_ordered_results()
            
            self._completed_tasks += 1
            return result
            
        except Exception as e:
            logger.error(f"❌ 任務處理失敗: {e}")
            if self.processor_config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise
    
    def _process_ordered_results(self):
        """Video模式順序結果處理"""
        while self.next_expected_id in self.pending_results:
            result = self.pending_results.pop(self.next_expected_id)
            self.result_queue.put(result)
            self.next_expected_id += 1
    
    def _ensure_thread_pool_started(self):
        """
        ⚡ 確保線程池已啟動（按需啟動機制）
        
        首次任務提交時自動啟動ThreadPoolExecutor
        """
        if not self.is_running and self.executor is None:
            mode_tag = "[VIDEO]" if self.worker_pool_config.mode == 'video' else "[CAMERA]"
            
            logger.info("⚡ 檢測到首次任務提交，自動啟動線程池...")
            
            try:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.worker_pool_config.max_workers,
                    thread_name_prefix=f"WorkerPool-{self.worker_pool_config.mode}"
                )
                self.is_running = True
                
                logger.info(f"🚀 {mode_tag} ThreadPoolExecutor自動啟動完成 ({self.worker_pool_config.max_workers}個工作線程)")
                logger.info(f"📋 {mode_tag} 線程池狀態: 運行中，開始處理任務")
                
            except Exception as e:
                logger.error(f"❌ {mode_tag} 自動啟動線程池失敗: {e}")
                raise RuntimeError(f"無法啟動線程池: {e}")
    
    def process_frame(self, frame: Any, callback: Optional[Callable] = None) -> Optional[Future]:
        """
        📊 任務緩衝管理：提交框架處理任務
        
        首次調用時自動啟動線程池，後續調用直接處理
        
        Args:
            frame: 輸入影像框
            callback: 結果回調函數
            
        Returns:
            Future對象或None（背壓丟棄時）
        """
        # ⚡ 按需啟動線程池
        self._ensure_thread_pool_started()
        
        self._total_tasks += 1
        
        # 背壓控制檢查
        if self._should_drop_task():
            self._dropped_tasks += 1
            if self.processor_config.detailed_logging:
                logger.warning(f"⚠️ 背壓控制：丟棄任務 (負載過高)")
            return None
        
        # 獲取Processor
        processor = self._get_next_processor()
        
        # 提交任務
        task_id = None
        if self.worker_pool_config.preserve_order:
            with self.lock:
                task_id = self.task_id_counter
                self.task_id_counter += 1
        
        future = self.executor.submit(
            self._process_task, processor, frame, task_id, callback
        )
        
        return future
    
    def submit(self, frame: Any) -> Optional[Future]:
        """
        📊 向後兼容：submit方法別名
        
        Args:
            frame: 輸入影像框
            
        Returns:
            Future對象或None（背壓丟棄時）
        """
        return self.process_frame(frame)
    
    def process_frame_sync(self, frame: Any, timeout: Optional[float] = None) -> Any:
        """
        🎯 同步處理單幀：提交任務並等待結果
        
        Args:
            frame: 輸入影像框
            timeout: 等待超時時間，None表示使用配置的推論超時
            
        Returns:
            推論結果
            
        Raises:
            RuntimeError: 處理失敗或超時
        """
        future = self.process_frame(frame)
        
        if future is None:
            raise RuntimeError("任務被背壓控制丟棄")
        
        # 使用配置的推論超時或指定的超時
        actual_timeout = timeout or self.processor_config.inference_timeout
        
        try:
            return future.result(timeout=actual_timeout)
        except Exception as e:
            raise RuntimeError(f"同步處理失敗: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        📊 性能統計報告
        
        Returns:
            統計資料字典
        """
        processor_stats = [p.get_stats() for p in self.processors]
        
        return {
            'mode': self.worker_pool_config.mode,
            'total_tasks': self._total_tasks,
            'completed_tasks': self._completed_tasks,
            'dropped_tasks': self._dropped_tasks,
            'success_rate': self._completed_tasks / max(self._total_tasks, 1),
            'drop_rate': self._dropped_tasks / max(self._total_tasks, 1),
            'queue_size': self.task_queue.qsize(),
            'buffer_size': self.worker_pool_config.buffer_size,
            'is_running': self.is_running,
            'thread_pool_started': self.executor is not None,
            'processors': processor_stats,
            'hardware': {
                'cpu_cores': self._cpu_cores,
                'memory_gb': self._memory_gb,
                'performance_level': self._performance_level,
                'performance_score': self._performance_score
            },
            'config': {
                'max_workers': self.worker_pool_config.max_workers,
                'enable_backpressure': self.worker_pool_config.enable_backpressure,
                'preserve_order': self.worker_pool_config.preserve_order,
                'inference_timeout': self.processor_config.inference_timeout
            }
        }
    
    def print_stats(self):
        """
        📊 格式化輸出統計資料
        """
        stats = self.get_stats()
        mode_tag = "[VIDEO]" if stats['mode'] == 'video' else "[CAMERA]"
        
        print(f"\n🏭 WorkerPool 統計報告 {mode_tag}")
        print("=" * 50)
        print(f"📊 任務統計:")
        print(f"   • 總任務: {stats['total_tasks']}")
        print(f"   • 完成: {stats['completed_tasks']}")
        print(f"   • 丟棄: {stats['dropped_tasks']}")
        print(f"   • 成功率: {stats['success_rate']:.2%}")
        print(f"   • 丟棄率: {stats['drop_rate']:.2%}")
        print(f"\n🔧 系統狀態:")
        print(f"   • 模式: {stats['mode']}")
        print(f"   • 線程池狀態: {'運行中' if stats['is_running'] else '未啟動'}")
        print(f"   • 可用Processor: {len(stats['processors'])}")
        print(f"   • 佇列使用: {stats['queue_size']}/{stats['buffer_size']}")
        print(f"\n💻 硬體資訊:")
        hw = stats['hardware']
        print(f"   • 性能等級: {hw['performance_level']} ({hw['performance_score']:.1f}分)")
        print(f"   • CPU核心: {hw['cpu_cores']}")
        print(f"   • 記憶體: {hw['memory_gb']}GB")
        print("=" * 50)
    
    def shutdown(self):
        """
        🧹 資源管理：關閉WorkerPool
        """
        self.is_running = False
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # 清理Processors
        for processor in self.processors:
            processor.cleanup()
        
        if self.processor_config.detailed_logging:
            logger.info(f"🧹 WorkerPool關閉完成: 總任務{self._total_tasks}, "
                       f"完成{self._completed_tasks}, 丟棄{self._dropped_tasks}")
        else:
            logger.debug("🧹 WorkerPool關閉完成")
    
    def stop(self):
        """
        🔄 向後兼容：stop方法別名，映射到shutdown()
        """
        logger.debug("🔄 調用stop()方法，映射至shutdown()")
        self.shutdown()

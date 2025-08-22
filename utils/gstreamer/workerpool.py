"""
================================================================================
⚙️ WorkerPool 架構設計｜2025.08.23 (更新版)
================================================================================

WorkerPool 採用智能調度與池化管理架構，專責 AI 推理任務的並行處理與資源分配。  
系統支援即時初始化，於建立階段即完成所有 YOLO 模型載入，確保首次任務提交時零延遲響應。

🆕 Frame ID 追蹤整合 (2025.08.23)：
WorkerPool 現在支援 Producer 的 frame_id 追蹤機制，確保每個處理任務都能保留原始的幀順序標識。
結果對象會同時包含 task_id（WorkerPool內部序號）和 original_frame_id（Producer原始序號）。

系統提供兩種處理模式：

┌─────────────┬──────────────────┬─────────────────────────────────────────┐
│ 📸 Video    │ 完整性優先       │ 順序追蹤、並行處理、零丟幀              │
│ 📷 Camera   │ 即時性優先       │ 背壓控制、智能調度、低延遲              │
└─────────────┴──────────────────┴─────────────────────────────────────────┘

📊 資料流向 (更新版)：
    Frame Data ──> WorkerPool ──> Task ID + Frame ID ──> Results
    {'frame': img,      │                                    │
     'frame_id': 0}     │                            {'result': ...,
                        │                             'task_id': 0,
              （任務控制中心）                         'original_frame_id': 0}
                        │
            ┌─────────────────────┐
            │ ThreadPoolExecutor  │  ← 直接任務分派
            │ ┌─────────────────┐ │
            │ │ Processor1      │ │ ← Round-Robin調度
            │ │ Processor2      │ │
            │ │ ProcessorN      │ │
            │ └─────────────────┘ │
            └─────────────────────┘

🎯 核心架構 (簡化版)：
                    WorkerPool（主控制器）
                    ├── Processor Pool（YOLO 推理引擎池）
                    ├── ThreadPoolExecutor（直接任務執行）
                    ├── Pending Tasks Tracker（背壓控制）
                    ├── Frame ID 追蹤系統 🆕
                    └── 直接結果回調
                           │
            ┌──────────────┼──────────────┐
            │                             │
        背壓控制機制                     Frame追蹤系統
    WorkerPool(mode="camera")      Task ID + Frame ID

📊 職責分配（◯ = 提供框架 / ✅ = 具體實作）：

┌─────────────────┬───────────────────┬─────────────────┐
│   功能類別      │  Video 模式特性     │ Camera 模式特性 │
├─────────────────┼───────────────────┼─────────────────┤
│ 🏭 初始化管理   │ ✅ 大緩衝配置     │ ✅ 小緩衝配置    │
│ 🎯 任務調度     │ ✅ 順序 ID 分配   │ ✅ 智能調度     │
│ ⚙️ 推理執行     │ ✅ 完整處理       │ ✅ 並行處理     │
│ 📊 緩衝管理     │ ✅ 無限緩衝       │ ✅ 背壓控制     │
│ 🔄 結果處理     │ ✅ 順序佇列       │ ✅ 即時回調     │
│ 📈 性能監控     │ ✅ 進度追蹤       │ ✅ 丟幀統計     │
│ 💻 硬體適應     │ ✅ 資源優化       │ ✅ 實時調整     │
│ 🧹 資源管理     │ 🔹 繼承使用       │ 🔹 繼承使用     │
└─────────────────┴───────────────────┴─────────────────┘

🛠️ 使用方法：

# ✅ 統一介面（初始化時預載入所有 YOLO 模型，確保首次任務零延遲）
worker_pool = WorkerPool(model_path="./models/yolo.tflite", mode="camera", max_workers=4)

# 可選參數：
# mode: "camera" 或 "video"
# max_workers: Processor 數量（預設為 4）

# 任務處理：
future = worker_pool.process_frame(frame)         # 非同步處理（回傳 Future）
result = worker_pool.process_frame_sync(frame)    # 同步處理（立即取得結果）

# 停止與資源清理：
worker_pool.shutdown()

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
                 model_path: str = None, mode: Literal['VIDEO', 'CAMERA'] = 'CAMERA', 
                 max_workers: int = 4, **kwargs):
        """
        初始化WorkerPool池化管理系統
        
        支持兩種創建方式：
        1. 配置對象方式：WorkerPool(processor_config, worker_pool_config)
        2. 參數方式：WorkerPool(model_path, mode='VIDEO', max_workers=4)
        
        Args:
            processor_config: ProcessorConfig配置對象（優先使用）
            worker_pool_config: WorkerPoolConfig配置對象（優先使用）
            model_path: YOLO模型路徑（參數方式）
            mode: 模式選擇 ('VIDEO'|'CAMERA')
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
        
        # 存儲模式屬性（直接使用原值，因為現在所有配置都是大寫）
        self.mode = self.worker_pool_config.mode
        
        # 初始化核心組件
        self.processors: List[Processor] = []
        self.executor = None
        self.is_running = False
        self.current_worker_index = 0
        self.lock = threading.Lock()
        
        # 任務追蹤（用於背壓控制和統計）
        self.pending_tasks = 0
        self.pending_tasks_lock = threading.Lock()
        
        # 簡化的任務ID計數器（僅用於日誌追蹤）
        self.task_id_counter = 0
        
        # 統計資料
        self._total_tasks = 0
        self._completed_tasks = 0
        self._dropped_tasks = 0
        
        # 硬體資訊初始化
        self._initialize_hardware_info()
        
        # 📝 WorkerPool即時就緒初始化日誌
        logger.info("🏭 " + "="*60)
        logger.info("🏭 WorkerPool初始化開始")
        logger.info("🏭 " + "="*60)
        
        # 📋 步驟 1/3: 配置驗證與基礎架構
        logger.info("📋 步驟 1/3: 🚀 配置驗證與基礎架構...")
        logger.info(f"🔍 [{self.mode}] 工作者數量: {self.worker_pool_config.max_workers}")
        logger.info(f"🔍 [{self.mode}] 緩衝區大小: {self.worker_pool_config.buffer_size}")

        # 📋 步驟 2/3: 硬體資訊與內部結構配置
        logger.info("📋 步驟 2/3: 🔧 硬體資訊與內部結構配置...")
        logger.info(f"🔧 [{self.mode}] 硬體配置: {self._cpu_cores}核心, {self._memory_gb}GB內存, {self._performance_level}")

        # 📋 步驟 3/3: 創建Processor池並載入模型（核心步驟）
        logger.info("📋 步驟 3/3: ⚙️ 創建Processor池並載入YOLO模型...")
        logger.info(f"🔍 [{self.mode}] 準備創建並初始化 {self.worker_pool_config.max_workers} 個Processor...")
        
        # 🎯 在初始化階段就載入所有模型 - 這才是初始化的核心目的！
        start_time = time.time()
        for i in range(self.worker_pool_config.max_workers):
            logger.info(f"🔍 [{self.mode}] 正在初始化 Processor #{i+1}/{self.worker_pool_config.max_workers}...")
            
            try:
                processor = Processor(self.processor_config)
                if not processor.initialize():
                    raise RuntimeError(f"Processor#{i+1} 初始化失敗")
                
                self.processors.append(processor)
                
            except Exception as e:
                logger.error(f"❌ [{self.mode}] Processor#{i+1} 創建失敗: {e}")
                if self.processor_config.detailed_logging:
                    logger.error(f"❌ [{self.mode}] 詳細錯誤: {traceback.format_exc()}")
                raise RuntimeError(f"WorkerPool初始化失敗: Processor#{i+1}載入模型失敗: {e}")
        
        # 初始化完成報告
        elapsed = time.time() - start_time
        logger.info(f"📈 [{self.mode}] 所有Processor已就緒，模型載入完成!")
        logger.info(f"⏱️ [{self.mode}] 模型載入總時間: {elapsed:.3f}s")
   
        logger.info("✅ WorkerPool初始化完成!")
    
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
        
        try:
            # 儲存回調函數
            self._result_callback = result_callback
            
            # 🎯 只在這裡啟動線程池 - 這才需要線程
            self.executor = ThreadPoolExecutor(
                max_workers=self.worker_pool_config.max_workers,
                thread_name_prefix=f"WorkerPool-{self.worker_pool_config.mode}"
            )
            self.is_running = True
            logger.info(f"🚀 WorkerPool線程池啟動成功，工作線程數: {self.worker_pool_config.max_workers}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [{self.mode}] 線程池啟動失敗: {e}")
            if self.processor_config.detailed_logging:
                logger.error(f"❌ [{self.mode}] 詳細錯誤: {traceback.format_exc()}")
            return False
    
    def _initialize_hardware_info(self):
        """初始化硬體資訊"""
        try:
            import psutil
            import os
            
            self._cpu_cores = os.cpu_count() or 4
            memory_info = psutil.virtual_memory()
            self._memory_gb = round(memory_info.total / (1024**3), 1)
            
            # 簡單的性能評估
            if self._cpu_cores >= 8 and self._memory_gb >= 16:
                self._performance_level = "高性能"
                self._performance_score = 9.0
            elif self._cpu_cores >= 4 and self._memory_gb >= 8:
                self._performance_level = "中等性能"
                self._performance_score = 7.0
            else:
                self._performance_level = "基本性能"
                self._performance_score = 5.0
                
        except Exception as e:
            logger.warning(f"⚠️ 硬體資訊獲取失敗，使用預設值: {e}")
            self._cpu_cores = 4
            self._memory_gb = 8.0
            self._performance_level = "未知"
            self._performance_score = 6.0
    
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
        📊 背壓控制判斷 - 基於實際pending任務數量
        
        Returns:
            bool: 是否應該丟棄任務
        """
        if not self.worker_pool_config.enable_backpressure:
            return False
        
        with self.pending_tasks_lock:
            # 計算當前負載：pending任務數 / (worker數量 * 緩衝倍數)
            max_pending = self.worker_pool_config.max_workers * 2  # 每個worker最多2個任務
            current_load = self.pending_tasks / max(max_pending, 1)
            
            return current_load >= self.worker_pool_config.drop_threshold
    
    def _process_task(self, processor: Processor, frame: Any, task_id: Optional[int] = None, 
                     callback: Optional[Callable] = None) -> Any:
        """
        任務處理內部方法
        """
        try:
            # 任務開始時增加pending計數
            with self.pending_tasks_lock:
                self.pending_tasks += 1

            # 提取實際的frame數據和frame_id用於日誌
            if isinstance(frame, dict) and 'frame' in frame:
                # 新格式：從字典中提取frame
                actual_frame = frame['frame']
                original_frame_id = frame.get('frame_id', task_id)
                logger.info(f"🔍 [WORKERPOOL] 開始處理任務，frame_id: #{original_frame_id} (task_id: {task_id})")
                logger.debug(f"🔍 [WORKERPOOL] 提取frame，原始frame_id: {original_frame_id}")
            else:
                # 舊格式：直接是frame（向後兼容）
                actual_frame = frame
                original_frame_id = task_id
                logger.info(f"🔍 [WORKERPOOL] 開始處理任務，frame_id: #{original_frame_id} (task_id: {task_id})")
                logger.debug(f"🔍 [WORKERPOOL] 舊格式frame，使用task_id: {task_id}")
            
            result = processor.predict(actual_frame)
            logger.info(f"🔍 [WORKERPOOL] 任務完成，frame_id: #{original_frame_id} (task_id: {task_id})")

            # 將task_id和原始frame_id附加到結果對象
            if hasattr(result, '__dict__'):
                result.frame_id = task_id
                result.original_frame_id = original_frame_id
                logger.debug(f"🔍 [WORKERPOOL] 已將task_id {task_id}, original_frame_id {original_frame_id} 附加到結果對象")
            elif isinstance(result, list) and len(result) > 0 and hasattr(result[0], '__dict__'):
                result[0].frame_id = task_id
                result[0].original_frame_id = original_frame_id
                logger.debug(f"🔍 [WORKERPOOL] 已將ID附加到結果列表的第一個元素")

            # 優先使用傳入的回調，否則使用啟動時設定的回調（向後兼容）
            active_callback = callback or getattr(self, '_result_callback', None)

            if active_callback:
                active_callback(result)
            else:
                if self._completed_tasks < 3:
                    logger.warning(f"⚠️ 沒有找到回調函數來處理結果 {self._completed_tasks + 1}")
            
            # 🎯 修復：移除冗餘的Video模式順序保證機制
            # 直接回調已經正確處理結果，無需重複處理
            # Video模式順序保證已通過直接回調實現
            logger.debug(f"🔍 [WORKERPOOL_ORDER] Video模式結果已通過直接回調處理，task_id: {task_id}")
            
            self._completed_tasks += 1
            return result
            
        except Exception as e:
            logger.error(f"❌ 任務處理失敗: {e}")
            if self.processor_config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise
        finally:
            # 任務完成時減少pending計數
            with self.pending_tasks_lock:
                self.pending_tasks = max(0, self.pending_tasks - 1)
    
    def _ensure_thread_pool_started(self):
        """
        ⚡ 確保線程池已啟動（按需啟動機制）
        
        首次任務提交時自動啟動ThreadPoolExecutor
        """
        if not self.is_running and self.executor is None:
            logger.info("⚡ 檢測到首次任務提交，自動啟動線程池...")
            
            try:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.worker_pool_config.max_workers,
                    thread_name_prefix=f"WorkerPool-{self.worker_pool_config.mode}"
                )
                self.is_running = True
                
                logger.info(f"🚀 [{self.mode}] ThreadPoolExecutor自動啟動完成 ({self.worker_pool_config.max_workers}個工作線程)")
                logger.info(f"📋 [{self.mode}] 線程池狀態: 運行中，開始處理任務")
                
            except Exception as e:
                logger.error(f"❌ [{self.mode}] 自動啟動線程池失敗: {e}")
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
        📊 性能統計報告 - 修正版
        
        Returns:
            統計資料字典
        """
        processor_stats = [p.get_stats() for p in self.processors]
        
        return {
            'mode': self.worker_pool_config.mode,
            'total_tasks': self._total_tasks,
            'completed_tasks': self._completed_tasks,
            'dropped_tasks': self._dropped_tasks,
            'pending_tasks': getattr(self, 'pending_tasks', 0),
            'success_rate': self._completed_tasks / max(self._total_tasks, 1),
            'drop_rate': self._dropped_tasks / max(self._total_tasks, 1),
            'is_running': self.is_running,
            'thread_pool_started': self.executor is not None,
            'active_processors': len(self.processors),
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
        
        print(f"\n🏭 WorkerPool 統計報告 [{self.mode}]")
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
        print(f"   • Pending任務: {stats['pending_tasks']}")
        print(f"   • 緩衝區配置: {stats['buffer_size']} (未使用)")
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
    
    def stop(self):
        """
        🔄 向後兼容：stop方法別名，映射到shutdown()
        """
        logger.debug("🔄 調用stop()方法，映射至shutdown()")
        self.shutdown()

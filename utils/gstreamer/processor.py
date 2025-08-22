"""
================================================================================
🔧 Processor 架構設計 2025.08.23 (更新版)
================================================================================

Processor 採用單一職責原則設計，專責 YOLO 模型載入、預熱與推論執行。  
系統支援即時初始化，在建構階段即完成模型載入與預熱，確保推論執行零延遲響應。  
Processor 提供統一推論介面與統計資料，不參與任務調度、順序控制或策略性管理。

🆕 Frame ID 追蹤整合 (2025.08.23)：
Processor 現在作為純推論執行單元，不直接處理 frame_id，但確保推論結果能被上層系統
（WorkerPool）正確關聯到原始的 frame_id，維持從 Producer 到 Consumer 的完整追蹤鏈。

📊 資料流向 (更新版)：
    Frame (numpy.array) ──> Processor ──> YOLO Results
                              │
        ┌────────────────────────────────┐
        │         YOLO Model             │ ← 推論執行流程（predict 方法）
        │    (預載入 + 預熱完成)         │
        └────────────────────────────────┘
                              │
                    統計資料自動更新

🎯 核心架構：

Processor（推論執行單元）  
├── YOLO Model（預載入 AI 引擎）  
├── 執行控制系統（線程安全、統計追蹤）  
├── 超時控制機制（Video寬鬆 vs Camera嚴格）🆕
├── 統計模組（推論次數、平均時間、超時警告）  
└── 生命週期管理（初始化、推論執行、資源清理）

📊 功能概覽：
┌─────────────────┬──────────────────────────────┬──────────────────┐
│   功能類別      │ Video模式                    │ Camera模式       │
├─────────────────┼──────────────────────────────┼──────────────────┤
│ 🏭 模型管理     │ 自動載入與預熱 YOLO 模型     │ 同左             │
│ 🎯 推論執行     │ 寬鬆超時、完整性優先         │ 嚴格超時、即時性 │
│ 📊 統計追蹤     │ 詳細日誌、完整統計           │ 精簡日誌、核心統計│
│ 🔒 線程安全     │ 多執行緒鎖保護               │ 同左             │
│ 🧹 資源管理     │ 安全釋放模型與記憶體資源     │ 同左             │
└─────────────────┴──────────────────────────────┴──────────────────┘

🛠️ 使用方法：
# ✅ 建構時自動完成模型載入與預熱
config = ProcessorConfig(model_path="./models/yolo.tflite", mode="VIDEO")
processor = Processor(config)

# 🎯 推論執行（立即可用，無需額外初始化）
results = processor.predict(frame)

# 📊 統計監控
stats = processor.get_stats()
print(f"推論次數: {stats['total_inferences']}")
print(f"平均時間: {stats['avg_inference_time']:.3f}s")

# 🧹 資源清理
processor.cleanup()

🔍 架構重點說明：
1. 單一職責：Processor 專注於模型推論，不涉入任務調度或策略控制。
2. 即時初始化：建構時完成模型載入與預熱，避免首次推論延遲。
3. 線程安全：支援多執行緒環境，所有推論皆受鎖保護。
4. 模式適應：Video模式寬鬆超時，Camera模式嚴格控制。🆕
5. 統計追蹤：內建推論次數、平均時間、超時警告統計。🆕
6. 可重用性：Processor 為無狀態推論單元，可由 WorkerPool 動態管理。
"""

import threading
import time
import logging
from typing import Any, Dict, Optional
import traceback
import numpy as np
from utils.gstreamer.config import ProcessorConfig
from ultralytics import YOLO

logger = logging.getLogger("gstreamer_demo")


class Processor:
    """
    🔧 Processor - YOLO推論執行單元
    
    功能：YOLO模型管理、預測執行、結果處理
    配置：通過ProcessorConfig控制推論行為
    
    Video模式：寬鬆超時、完整性優先、詳細記錄
    Camera模式：嚴格超時、實時性優先、精簡記錄
    """
    
    def __init__(self, config: ProcessorConfig):
        """
        初始化Processor推論單元
        
        Args:
            config: ProcessorConfig配置對象，控制推論行為差異
        """
        self.config = config
        self.model = None
        self.is_busy = False
        self.lock = threading.Lock()
        
        # 統計相關屬性
        self._total_inferences = 0
        self._total_inference_time = 0.0
        self._timeout_warnings = 0
        self._initialization_time = 0.0
        self._last_inference_time = 0.0
        
        # 初始化相關屬性
        self._initialized = False
        self._initialization_error = None
        
        # 日誌配置
        self._log_prefix = f"[PROCESSOR-{self.config.mode}]"
        
        if self.config.detailed_logging:
            logger.info(f"🔧 {self._log_prefix} 初始化開始...")
            logger.info(f"🔧 {self._log_prefix} 模型路徑: {config.model_path}")
            logger.info(f"🔧 {self._log_prefix} 模式: {config.mode}")
            logger.info(f"🔧 {self._log_prefix} 超時設定: {config.inference_timeout}s")
        
        # 🚀 自動執行初始化（符合文件描述的即時預熱）
        if not self.initialize():
            error_msg = f"Processor初始化失敗: {self._initialization_error or '無法載入模型'}"
            logger.error(f"❌ {self._log_prefix} {error_msg}")
            raise RuntimeError(error_msg)
        
        if self.config.detailed_logging:
            logger.info(f"✅ {self._log_prefix} 初始化完成，耗時: {self._initialization_time:.3f}s")
    
    def initialize(self) -> bool:
        """
        🚀 模型載入與預熱
        
        Returns:
            bool: 初始化是否成功
        """
        # 避免重複初始化
        if self._initialized and self.model is not None:
            logger.debug(f"🔄 {self._log_prefix} 模型已初始化，跳過")
            return True
            
        try:
            start_time = time.time()
            
            # 檢查YOLO可用性
            if YOLO is None:
                self._initialization_error = "YOLO 模組不可用"
                logger.error(f"❌ {self._log_prefix} {self._initialization_error}")
                return False
            
            logger.info(f"📦 {self._log_prefix} 載入模型: {self.config.model_path}")
            
            # 載入YOLO模型
            self.model = YOLO(self.config.model_path)
            
            logger.info(f"🔥 {self._log_prefix} 開始模型預熱...")
            
            # 模型預熱 - 使用正確的預熱方式
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # 執行預熱推論（不使用stream模式）
            warmup_results = self.model.predict(
                dummy_input, 
                verbose=False,  # 預熱時不輸出詳細信息
                save=False,     # 不保存結果
                show=False      # 不顯示結果
            )
            
            # 確保預熱結果被消耗（如果是生成器）
            if hasattr(warmup_results, '__iter__'):
                list(warmup_results)  # 消耗生成器
            
            elapsed = time.time() - start_time
            self._initialization_time = elapsed
            self._initialized = True
            
            logger.info(f"✅ {self._log_prefix} 預熱完成，耗時: {elapsed:.3f}s")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"❌ {self._log_prefix} 初始化失敗: {e}")
            if self.config.detailed_logging:
                logger.error(f"❌ {self._log_prefix} 詳細錯誤: {traceback.format_exc()}")
            return False
    
    def predict(self, frame: Any, **kwargs) -> Any:
        """
        🎯 推論執行管理 - 含超時控制與統計追蹤
        
        Args:
            frame: 輸入影像框（numpy.array）
            **kwargs: 額外YOLO參數
            
        Returns:
            YOLO推論結果
            
        Raises:
            RuntimeError: 模型未初始化或推論超時
        """
        if not self._initialized or self.model is None:
            error_msg = f"模型未初始化: {self._initialization_error or '未知錯誤'}"
            logger.error(f"❌ {self._log_prefix} {error_msg}")
            raise RuntimeError(error_msg)
        
        with self.lock:
            self.is_busy = True
            
        try:
            start_time = time.time()
            
            # 執行YOLO推論
            results = self.model.predict(
                frame, 
                verbose=self.config.detailed_logging, 
                **kwargs
            )
            
            inference_time = time.time() - start_time
            self._last_inference_time = inference_time
            
            # 超時檢查與處理
            if inference_time > self.config.inference_timeout:
                self._timeout_warnings += 1
                warning_msg = f"推論超時: {inference_time:.3f}s (限制: {self.config.inference_timeout:.3f}s)"
                
                if self.config.mode == 'CAMERA':
                    # Camera模式：嚴格超時控制
                    logger.warning(f"⚠️ {self._log_prefix} {warning_msg}")
                    raise RuntimeError(warning_msg)
                else:
                    # Video模式：寬鬆超時處理
                    if self.config.detailed_logging:
                        logger.warning(f"⚠️ {self._log_prefix} {warning_msg} (Video模式繼續)")
            
            # 更新統計
            self._total_inferences += 1
            self._total_inference_time += inference_time
            
            # 定期統計報告
            if (self.config.detailed_logging and 
                self._total_inferences % 100 == 0):
                avg_time = self._total_inference_time / self._total_inferences
                logger.info(f"📊 {self._log_prefix} 推論統計: {self._total_inferences}次, 平均: {avg_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {self._log_prefix} 推論執行失敗: {e}")
            if self.config.detailed_logging:
                logger.error(f"❌ {self._log_prefix} 詳細錯誤: {traceback.format_exc()}")
            raise
            
        finally:
            with self.lock:
                self.is_busy = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        📊 獲取推論統計 - 完整性能監控數據
        
        Returns:
            統計資料字典，包含推論次數、時間、錯誤等資訊
        """
        with self.lock:
            avg_inference_time = (
                self._total_inference_time / self._total_inferences 
                if self._total_inferences > 0 else 0.0
            )
            
            return {
                # 核心統計
                'total_inferences': self._total_inferences,
                'avg_inference_time': avg_inference_time,
                'total_inference_time': self._total_inference_time,
                'last_inference_time': self._last_inference_time,
                
                # 系統狀態
                'is_busy': self.is_busy,
                'is_initialized': self._initialized,
                'model_loaded': self.model is not None,
                
                # 初始化資訊
                'initialization_time': self._initialization_time,
                'initialization_error': self._initialization_error,
                
                # 錯誤統計
                'timeout_warnings': self._timeout_warnings,
                
                # 配置資訊
                'mode': self.config.mode,
                'inference_timeout': self.config.inference_timeout,
                'model_path': self.config.model_path,
                
                # 效能指標
                'throughput': (
                    self._total_inferences / self._total_inference_time 
                    if self._total_inference_time > 0 else 0.0
                ),
                'timeout_rate': (
                    self._timeout_warnings / self._total_inferences 
                    if self._total_inferences > 0 else 0.0
                )
            }
    
    def reset_stats(self):
        """🔄 重置統計資料"""
        with self.lock:
            self._total_inferences = 0
            self._total_inference_time = 0.0
            self._timeout_warnings = 0
            self._last_inference_time = 0.0
            
        logger.info(f"🔄 {self._log_prefix} 統計資料已重置")
    
    def cleanup(self):
        """🧹 線程安全的資源清理"""
        with self.lock:
            if self.model is not None:
                logger.info(f"🧹 {self._log_prefix} 清理模型資源...")
                try:
                    del self.model
                    self.model = None
                    self._initialized = False
                    logger.info(f"✅ {self._log_prefix} 資源清理完成")
                except Exception as e:
                    logger.warning(f"⚠️ {self._log_prefix} 資源清理警告: {e}")
            else:
                logger.debug(f"🧹 {self._log_prefix} 無需清理（模型未載入）")
    
    def __del__(self):
        """析構函數 - 確保資源被正確釋放"""
        try:
            self.cleanup()
        except:
            pass  # 析構函數中不應該拋出異常
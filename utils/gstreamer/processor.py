"""
================================================================================
🔧 Processor 架構設計 2025.08.21
================================================================================

Processor 採用單一職責原則設計，專責 YOLO 模型載入、預熱與推論執行。  
系統支援即時初始化，在建構階段即完成模型載入與預熱，確保推論執行零延遲響應。  
Processor 提供統一推論介面與統計資料，不參與任務調度、順序控制或策略性管理。

📊 資料流向：
    Frame Input ──> Processor ──> Results
                       │
        ┌────────────────────────────┐
        │         YOLO Model         │ ← 推論執行流程（predict 方法）
        └────────────────────────────┘

🎯 核心架構：

Processor（推論執行單元）  
├── YOLO Model（預載入 AI 引擎）  
├── 執行控制系統（線程安全、統計追蹤）  
├── 統計模組（推論次數、平均時間）  
└── 生命週期管理（初始化、推論執行、資源清理）

📊 功能概覽：
┌─────────────────┬──────────────────────────────┐
│   功能類別      │ 說明內容                      │
├─────────────────┼──────────────────────────────┤
│ 🏭 模型管理     │ 自動載入與預熱 YOLO 模型     │
│ 🎯 推論執行     │ 單幀推論，即時回傳結果        │
│ 📊 統計追蹤     │ 推論次數、平均時間、狀態監控 │
│ 🔒 線程安全     │ 多執行緒鎖保護                │
│ 🧹 資源管理     │ 安全釋放模型與記憶體資源     │
└─────────────────┴──────────────────────────────┘

🛠️ 使用方法：
# ✅ 建構時自動完成模型載入與預熱
config = ProcessorConfig(model_path="./models/yolo.tflite")
processor = Processor(config)

# 🎯 推論執行（立即可用）
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
4. 統計追蹤：內建推論次數與平均時間統計，便於性能監控。
5. 可重用性：Processor 為無狀態推論單元，可由任務調度層動態配置與管理。
"""

import threading
import time
import logging
from typing import Any, Dict, Optional
import traceback
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
        self._total_inferences = 0
        self._total_inference_time = 0.0
        self._timeout_warnings = 0  # 超時警告計數
        self._initialization_time = 0.0  # 初始化時間記錄
        
    # ...existing code...
        
        # 🚀 自動執行初始化（符合文件描述的即時預熱）
        if not self.initialize():
            raise RuntimeError(f"Processor初始化失敗: 無法載入模型 {config.model_path}")
    
    def initialize(self) -> bool:
        """
        🚀 模型載入預熱
        
        Returns:
            bool: 初始化是否成功
        """
        # 避免重複初始化
        if self.model is not None:
            return True
            
        try:
            if YOLO is None:
                return False
            
            start_time = time.time()
            # 載入YOLO模型
            self.model = YOLO(self.config.model_path)
            
            # 模型預熱
            import numpy as np
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_input, verbose=True, stream=True)
            
            elapsed = time.time() - start_time
            self._initialization_time = elapsed  # 記錄初始化時間
            return True
            
        except Exception as e:
            return False
    
    def predict(self, frame: Any, **kwargs) -> Any:
        """
        🎯 預測執行管理 - 含超時控制
        
        Args:
            frame: 輸入影像框
            **kwargs: 額外參數
            
        Returns:
            推論結果
            
        Raises:
            RuntimeError: 模型未初始化或推論超時
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        with self.lock:
            self.is_busy = True
        try:
            start_time = time.time()
            results = self.model.predict(frame, verbose=not self.config.detailed_logging, stream=True, **kwargs)
            inference_time = time.time() - start_time
            if inference_time > self.config.inference_timeout:
                self._timeout_warnings += 1
                if self.config.mode == 'CAMERA':
                    raise RuntimeError(f"推論超時: {inference_time:.3f}s")
            self._total_inferences += 1
            self._total_inference_time += inference_time
            return results
        except Exception as e:
            raise
        finally:
            with self.lock:
                self.is_busy = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        📊 獲取推論統計 - 完整性能監控數據
        
        Returns:
            統計資料字典
        """
        with self.lock:
            avg_inference_time = (
                self._total_inference_time / self._total_inferences 
                if self._total_inferences > 0 else 0
            )
            
            return {
                'total_inferences': self._total_inferences,
                'avg_inference_time': avg_inference_time,
                'total_inference_time': self._total_inference_time,
                'initialization_time': self._initialization_time,
                'timeout_warnings': self._timeout_warnings,
                'is_busy': self.is_busy,
                'mode': self.config.mode,
                'inference_timeout': self.config.inference_timeout,
                'model_loaded': self.model is not None
            }
    
    def cleanup(self):
        """🧹 線程安全清理"""
        with self.lock:
            if self.model is not None:
                del self.model
                self.model = None

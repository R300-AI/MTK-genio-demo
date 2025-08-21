"""
================================================================================
🔧 Processor 架構設計
================================================================================
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
        
        # 📝 Processor初始化日誌 - 統一格式
        mode_tag = "[VIDEO]" if config.mode == 'video' else "[CAMERA]"
        logger.info("🔧 " + "="*60)
        logger.info("🔧 Processor初始化開始")
        logger.info("🔧 " + "="*60)
        logger.info("📋 步驟 1/2: 🚀 YOLO模型配置...")
        logger.info(f"🔍 {mode_tag} 模型路徑: {config.model_path}")
        logger.info(f"🔍 {mode_tag} 推論超時: {config.inference_timeout}s")
        
        if config.mode == 'video':
            logger.info("🎯 [VIDEO] 模式設定: 完整性優先")
            logger.info("📊 [VIDEO]   - 詳細記錄: 開啟")
            logger.info("📊 [VIDEO]   - 超時策略: 寬鬆 (確保完整處理)")
            logger.info("📊 [VIDEO]   - 模式特性: 順序保證、無丟幀")
        else:
            logger.info("🎯 [CAMERA] 模式設定: 實時性優先") 
            logger.info("📊 [CAMERA]   - 詳細記錄: 精簡")
            logger.info("📊 [CAMERA]   - 超時策略: 嚴格 (確保即時響應)")
            logger.info("📊 [CAMERA]   - 模式特性: 背壓控制、低延遲")
    
    def initialize(self) -> bool:
        """
        🚀 模型載入預熱
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if YOLO is None:
                logger.error("❌ YOLO未安裝，無法初始化Processor")
                return False
            
            start_time = time.time()
            mode_tag = "[VIDEO]" if self.config.mode == 'video' else "[CAMERA]"
            
            logger.info("📋 步驟 2/2: 🚀 YOLO模型載入與預熱...")
            logger.info(f"🔍 {mode_tag} 正在載入YOLO模型...")
            
            # 載入YOLO模型
            self.model = YOLO(self.config.model_path)
            logger.info(f"✅ {mode_tag} 模型載入成功")
            
            logger.info(f"🔥 {mode_tag} 執行預熱推論...")
            
            # 模型預熱
            import numpy as np
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_input, verbose=False, stream=True)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ {mode_tag} 預熱完成")
            
            if self.config.mode == 'video':
                logger.info("⏱️ [VIDEO] 初始化時間: {:.3f}s (完整載入)".format(elapsed))
                logger.info("📈 [VIDEO] 推論準備: 完整性模式就緒")
            else:
                logger.info("⏱️ [CAMERA] 初始化時間: {:.3f}s (快速載入)".format(elapsed))
                logger.info("📈 [CAMERA] 推論準備: 實時模式就緒")
                
            logger.info("✅ Processor初始化完成!")
            logger.info("🔧 " + "="*60)
                
            return True
            
        except Exception as e:
            mode_tag = "[VIDEO]" if self.config.mode == 'video' else "[CAMERA]"
            logger.error(f"❌ {mode_tag} Processor初始化失敗: {e}")
            if self.config.detailed_logging:
                logger.error(f"❌ {mode_tag} 詳細錯誤: {traceback.format_exc()}")
            return False
    
    def predict(self, frame: Any, **kwargs) -> Any:
        """
        🎯 預測執行管理
        
        Args:
            frame: 輸入影像框
            **kwargs: 額外參數
            
        Returns:
            推論結果
        """
        if self.model is None:
            raise RuntimeError("模型未初始化")
        
        with self.lock:
            self.is_busy = True
            
        try:
            start_time = time.time()
            
            # 執行YOLO推論
            results = self.model.predict(frame, verbose=not self.config.detailed_logging, stream=True, **kwargs)
            inference_time = time.time() - start_time
            
            # 更新統計
            self._total_inferences += 1
            self._total_inference_time += inference_time
            
            # 根據mode記錄不同詳細程度
            if self.config.detailed_logging:
                logger.debug(f"🎯 推論完成: {inference_time:.3f}s, 總計: {self._total_inferences}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 推論執行失敗: {e}")
            if self.config.detailed_logging:
                logger.error(f"詳細錯誤: {traceback.format_exc()}")
            raise
            
        finally:
            with self.lock:
                self.is_busy = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        📊 獲取推論統計
        
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
                'is_busy': self.is_busy,
                'mode': self.config.mode,
                'inference_timeout': self.config.inference_timeout
            }
    
    def cleanup(self):
        """🧹 線程安全清理"""
        with self.lock:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.config.detailed_logging:
                logger.info(f"🧹 Processor清理完成, 總推論: {self._total_inferences}")

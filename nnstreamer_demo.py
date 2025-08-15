#!/usr/bin/env python3
"""
NNStreamer YOLO 實現 - 使用者自定義推論器示例
適用於 MediaTek Genio 平台的高效AI推論管道

使用者可以修改 YOLOInterpreter 來自定義模型推論行為
"""

import asyncio
import argparse
import numpy as np
import warnings
from ultralytics import YOLO

# 導入核心模組
from utils.nnstreamer import NNStreamer, logger

# 抑制 Ultralytics 警告
warnings.filterwarnings("ignore")

class YOLOInterpreter:
    """
    Ultralytics YOLO 解釋器封裝類別
    
    這個類別可以由使用者自由修改來：
    - 更改推論參數
    - 自定義前處理/後處理
    - 實現不同的視覺化效果
    - 添加自定義邏輯
    """
    
    def __init__(self, model_path: str):
        try:
            # 載入模型
            self.model = YOLO(model_path, task="detect")
            
            # 預熱模型 - 使用隨機圖像進行一次推論
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_image, verbose=False)
            
            logger.info(f"YOLO 模型載入成功: {model_path}")
            logger.info("使用 YOLO 預設參數")
            
        except Exception as e:
            logger.error(f"YOLO 模型載入失敗: {e}")
            raise

    def inference(self, frame: np.ndarray):
        """
        執行推論並返回 YOLO Results 對象
        
        使用者可以修改這個方法來：
        - 調整推論參數 (conf, iou, imgsz 等)
        - 添加前處理步驟
        - 實現批次推論
        """
        try:
            results = self.model.predict(frame, verbose=False, save=False, show=False)
            return results[0]
            
        except Exception as e:
            logger.error(f"YOLO 推論執行失敗: {e}")
            raise

    def visualize(self, frame: np.ndarray, yolo_result):
        """
        視覺化檢測結果
        
        使用者可以修改這個方法來：
        - 自定義繪圖樣式
        - 添加額外資訊顯示
        - 實現不同的視覺化效果
        """
        try:
            # 使用 YOLO 內建的 plot 方法繪製檢測結果
            annotated_frame = yolo_result.plot(show=False, save=False)
            return annotated_frame
            
        except Exception as e:
            logger.error(f"視覺化失敗: {e}")
            return frame  # 回退到原始幀



def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="NNStreamer - Ultralytics YOLO 串流推論系統")
    parser.add_argument("--model", type=str, default="./models/yolov8n_float32.tflite", help="YOLO模型路徑")
    parser.add_argument("--input", type=str, default="./data/video.mp4", help="輸入來源")
    parser.add_argument("--workers", type=int, default=4, help="最大工作者數量")
    parser.add_argument("--queue_size", type=int, default=32, help="隊列大小")
    parser.add_argument("--no_display", action="store_true", help="禁用視覺輸出")
    args = parser.parse_args()

    # 創建 NNStreamer 實例，傳入 YOLOInterpreter 類別
    nnstreamer = NNStreamer(
        interpreter_class=YOLOInterpreter,
        model_path=args.model,
        input_source=args.input,
        max_workers=args.workers,
        queue_size=args.queue_size,
        display_output=not args.no_display
    )
    
    # 運行系統
    try:
        asyncio.run(nnstreamer.run())
    except KeyboardInterrupt:
        logger.info("程式被使用者中斷")
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
    finally:
        logger.info("程式結束")


if __name__ == "__main__":
    main()
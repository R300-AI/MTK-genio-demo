#!/usr/bin/env python3
import asyncio, cv2, argparse
import numpy as np
import warnings
from ultralytics import YOLO
from utils.nnstreamer import NNStreamer, logger

# 抑制 Ultralytics 警告
warnings.filterwarnings("ignore")

class CustomInferenceExecutor:
    """
    自定義推論執行器
    
    必要方法：
    - __init__(model_path: str) - 模型初始化
    - inference(input_frame: np.ndarray) - 推論邏輯
    - visualize(input_frame, model_output) - 視覺化
    
    可替換為任何 AI 框架：TensorFlow, PyTorch, ONNX 等
    """
    
    def __init__(self, model_path: str):
        """
        模型初始化
        """
        try:
            self.model = YOLO(model_path, task="detect")
            
            # 模型暖機避免首次延遲
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_input, verbose=False)
            
            logger.info(f"YOLO Model Loaded Successfully: {model_path}")
            
        except Exception as e:
            logger.error(f"FAILED: YOLO Model Loading Error: {e}")
            raise

    def inference(self, input_frame: np.ndarray):
        """
        核心推論過程
        """
        try:
            inference_results = self.model.predict(input_frame, verbose=False, save=False, show=False)
            return inference_results[0]
            
        except Exception as e:
            logger.error(f"FAILED: YOLO Inference Execution Error: {e}")
            raise

    def visualize(self, input_frame: np.ndarray, model_output):
        """
        輸出辨識結果結果
        """
        try:
            annotated_output = model_output.plot(show=False, save=False)
            return cv2.resize(annotated_output, (960, 540))
            
        except Exception as e:
            logger.error(f"FAILED: Visualization Error: {e}")
            return cv2.resize(input_frame, (960, 540))

def main():
    parser = argparse.ArgumentParser(description="NNStreamer - Ultralytics YOLO 串流推論系統")
    parser.add_argument("--model", type=str, default="./models/yolov8n_float32.tflite", help="YOLO模型路徑")
    parser.add_argument("--input", type=str, default="./data/video.mp4", help="輸入來源 (0=攝像頭, 檔案路徑=影片)")
    parser.add_argument("--workers", type=int, default=4, help="最大工作者數量")
    parser.add_argument("--queue_size", type=int, default=32, help="隊列大小")
    parser.add_argument("--target_fps", type=float, default=24.0, help="目標幀率")
    parser.add_argument("--no_display", action="store_true", help="禁用視覺輸出")
    args = parser.parse_args()

    input_source = args.input
    if input_source.isdigit():
        input_source = int(input_source)

    streaming_processor = NNStreamer(
        executor=CustomInferenceExecutor,
        model_path=args.model,
        input_source=input_source,
        max_workers=args.workers,
        max_queue_length=args.queue_size,
        display_output=not args.no_display,
        target_fps=args.target_fps
    )
    
    try:
        asyncio.run(streaming_processor.run())
    except KeyboardInterrupt:
        logger.info("Program Interrupted by User - Shutting Down")
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Program Execution Failed: {e}")
    finally:
        logger.info("Program Terminated Successfully")


if __name__ == "__main__":
    main()
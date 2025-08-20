import argparse
import logging
import time
from utils.gstreamer.producer import Producer
from utils.gstreamer.processor import WorkerPool
from utils.gstreamer.consumer import Consumer
from utils.gstreamer.pipeline import create_pipeline

logger = logging.getLogger('gstreamer_demo')

def main():
    parser = argparse.ArgumentParser(description="Hetegeneous Integrated Chip Inference Demo")
    parser.add_argument(
        "video_path",
        type=str,
        nargs='?',
        default="./data/video.mp4",
        help=(
            "Defines the input source for frame acquisition via OpenCV VideoCapture. "
            "This parameter supports camera indices (e.g., 0), RTSP stream URLs, or file paths to local video assets."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/yolov8n-pose_float32.tflite",
        help=(
            "Path to the YOLO model file for inference. "
            "Supports .tflite format. Default is './models/yolov8n-pose_float32.tflite'."
        )
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help=(
            "Maximum number of concurrent processing workers. "
            "Controls parallel inference threads."
        )
    )
    args = parser.parse_args()
    if (isinstance(args.video_path, str) and (args.video_path.isdigit() or
        args.video_path.lower().startswith(('rtsp://', 'rtmp://')))):
        mode = 'camera'
    else:
        mode = 'video'

    producer = Producer(args.video_path, mode=mode)
    worker_pool = WorkerPool(args.model_path, max_workers=args.max_workers)
    consumer = Consumer(
        window_name="Hetegeneous Integrated Chip Inference Demo", 
        display_size=(720, 480), 
        fps=30
    )

    pipeline = create_pipeline(
        producer=producer,   
        worker_pool=worker_pool,
        consumer=consumer
    )

    print("[系統] Pipeline 執行中...")
    pipeline.run()
    print("[系統] Pipeline 執行結束")

if __name__ == "__main__":
    main()
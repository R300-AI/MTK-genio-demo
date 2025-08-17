import argparse
import logging
from utils.gstreamer.producer import Producer
from utils.gstreamer.processor import WorkerPool
from utils.gstreamer.consumer import Consumer
from utils.gstreamer.pipeline import Pipeline
from utils.gstreamer.monitor import Monitor
from utils.gstreamer.balancer import Balancer

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

def main():
    logger.info("Starting GStreamer Demo")
    parser = argparse.ArgumentParser(description="Hetegeneous Integrated Chip Inference Demo")
    parser.add_argument(
        "--video_path",
        type=str,
        default="./data/video.mp4",#"./data/video.mp4"
        help=(
            "Defines the input source for frame acquisition via OpenCV VideoCapture. "
            "This parameter supports camera indices (e.g., 0), RTSP stream URLs, or file paths to local video assets."
        )
    )
    parser.add_argument("--model_path", type=str, default="./models/yolov8n_float32.tflite")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of concurrent processing workers")
    parser.add_argument("--disable_balancer", action="store_true", help="Disable dynamic frame rate balancing")
    parser.add_argument("--enable_backpressure_logging", action="store_true", help="Enable detailed backpressure logging")
    args = parser.parse_args()

    monitor = Monitor()
    
    # 建立 Balancer（預設啟用，除非明確停用）
    balancer = None
    if not args.disable_balancer:
        balancer = Balancer()
        logger.info("[MAIN] Balancer enabled for dynamic rate control")
    else:
        logger.info("[MAIN] Balancer disabled by user")
    
    producer = Producer(args.video_path, monitor=monitor, balancer=balancer)
    worker_pool = WorkerPool(args.model_path, monitor=monitor, max_workers=args.max_workers, balancer=balancer)
    consumer = Consumer(window_name="Hetegeneous Integrated Chip Inference Demo", monitor=monitor, display_size = (720, 480))

    pipeline = Pipeline(
        producer=producer,
        worker_pool=worker_pool,
        consumer=consumer,
        monitor=monitor,
        balancer=balancer
    )

    logger.info("========【Starting pipeline execution】========")
    pipeline.run()
    logger.info("========【Pipeline execution completed】========")

if __name__ == "__main__":
    main()
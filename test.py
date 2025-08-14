
import cv2, argparse
import multiprocessing
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="YOLO Multiprocessing Inference")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
parser.add_argument("--max_workers", type=int, default=6, help="Number of inference workers")
args = parser.parse_args()

multiprocessing.set_start_method('spawn', force=True)
frame_queue = multiprocessing.Queue(maxsize=100)

def preprocess(frame_queue):
    print("[讀取器] 啟動")
    cap = cv2.VideoCapture(args.video_path)
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((index, frame))
        index += 1
    for _ in range(args.max_workers):
        frame_queue.put((None, None))
    cap.release()
    print("[讀取器] 結束")

def inference_worker(worker_id, frame_queue):
    print(f"[推理器-{worker_id}] 啟動")
    model = YOLO(args.tflite_model, task="detect")
    while True:
        index, frame = frame_queue.get()
        if index is None and frame is None:
            print(f"[推理器-{worker_id}] 結束")
            break
        result = model.predict(frame, verbose=False)[0]
        print(f"[推理器-{worker_id}] {index} {result.plot().shape}")

if __name__ == "__main__":
    workers = [multiprocessing.Process(target=inference_worker, args=(i, frame_queue)) for i in range(args.max_workers)]
    for w in workers:
        w.start()
    preprocess(frame_queue)
    for w in workers:
        w.join()
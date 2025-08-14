from ultralytics import YOLO
import cv2, asyncio, time, argparse

async def preprocess(input_queue, cap):
    while cap.isOpened():
        ret, frame = cap.read
        if not ret:
            break
        else:
            await input_queue.put(frame)
    cap.release()

async def inference(input_queue, output_queue, model):
    while True:
        frame = await input_queue.get()
        if frame is not None:
            result = model.predict(frame, stream=False)[0]
            await output_queue.put(result)

async def postprocess(output_queue):
    while True:
        result = await output_queue.get()
        if result is not None:
            result = result.plot()
            cv2.imshow("Ultralytics Async", cv2.resize(result , (720, 480)))
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    
# ------------------ 主流程（平鋪 + 最小 async 啟動器） ------------------
parser = argparse.ArgumentParser(description="Ultralytics YOLO 非同步串流推論（Async Streaming Demo）")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video_path}")

input_queue = asyncio.Queue(maxsize=30)
output_queue = asyncio.Queue(maxsize=30)

print("[main] 初始化模型一次（YOLO / TFLite / DLA）...")
model = YOLO(args.tflite_model, task="detect")
print("[main] 模型初始化完成")

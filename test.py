from ultralytics import YOLO
import cv2, argparse
import asyncio

parser = argparse.ArgumentParser(description="YOLO Async Inference with asyncio")

parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
parser.add_argument("--num_workers", type=int, default=6, help="Number of inference workers")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
model = YOLO(args.tflite_model, task="detect")

frame_queue = asyncio.Queue(maxsize=100)

async def preprocess():
    print("[讀取器] 啟動")
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        await frame_queue.put((index, frame))
        index += 1
    # 放入 None 給每個 inference worker
    for _ in range(args.num_workers):
        await frame_queue.put(None)
    cap.release()
    print("[讀取器] 結束")

async def inference(worker_id):
    print(f"[推理器-{worker_id}] 啟動")
    while True:
        index, frame = await frame_queue.get()
        if frame is None:
            await result_queue.put(None)
            break
        result = model.predict(frame, verbose=False)[0]
        print(f"[推理器-{worker_id}] {index} {result.plot().shape}")

async def main():
    tasks = [asyncio.create_task(preprocess())]
    for i in range(args.num_workers):
        tasks.append(asyncio.create_task(inference(i)))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

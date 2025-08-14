from ultralytics import YOLO
import cv2, argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="YOLO Async Inference with asyncio")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
parser.add_argument("--max_workers", type=int, default=1, help="Number of predict workers")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
model = YOLO(args.tflite_model, task="detect")


frame_queue = asyncio.Queue(maxsize=100)
result_queue = asyncio.Queue(maxsize=100)
executor = ThreadPoolExecutor(max_workers=max(1, args.max_workers))

async def reader():
    print("[讀取器] 啟動")
    loop = asyncio.get_running_loop()
    while True:
        ret, frame = await loop.run_in_executor(executor, cap.read)
        if not ret:
            break
        await frame_queue.put(frame)
    for _ in range(args.max_workers):
        await frame_queue.put(None)
    cap.release()
    print("[讀取器] 結束")

async def predictor(worker_id):
    loop = asyncio.get_running_loop()
    print(f"[推理器] 啟動")
    while True:
        frame = await frame_queue.get()
        if frame is None:
            await result_queue.put(None)
            print(f"[推理器] 收到結束訊號，離開。")
            break
        result = await loop.run_in_executor(executor, lambda: model.predict(frame, verbose=False)[0])
        await result_queue.put(result)

async def displayer():
    print("[顯示器] 啟動")
    loop = asyncio.get_running_loop()
    while True:
        result = await result_queue.get()
        if result is None:
            print(f"[顯示器] 收到所有結束訊號，離開。")
            break
        img = await loop.run_in_executor(executor, lambda: cv2.resize(result.plot(), (720, 480)))
        await loop.run_in_executor(executor, lambda: cv2.imshow("Ultralytics Async Inference", img))
        key = await loop.run_in_executor(executor, lambda: cv2.waitKey(1))
    cv2.destroyAllWindows()
    print(f"[顯示器] 結束")

async def main():
    tasks = [asyncio.create_task(reader())]
    for i in range(args.max_workers):
        tasks.append(asyncio.create_task(predictor(i)))
    tasks.append(asyncio.create_task(displayer()))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
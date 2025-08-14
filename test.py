
import cv2
import argparse
import asyncio
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLO Asyncio Inference")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
parser.add_argument("--max_workers", type=int, default=6, help="Number of inference workers")
args = parser.parse_args()

async def preprocess(frame_queue: asyncio.Queue):
    print("[讀取器] 啟動")
    cap = cv2.VideoCapture(args.video_path)
    index = 0
    loop = asyncio.get_running_loop()
    while True:
        # OpenCV I/O is blocking, so run in thread
        ret, frame = await asyncio.to_thread(cap.read)
        if not ret:
            break
        await frame_queue.put((index, frame))
        index += 1
    for _ in range(args.max_workers):
        await frame_queue.put((None, None))
    cap.release()
    print("[讀取器] 結束")

async def inference_worker(worker_id: int, frame_queue: asyncio.Queue):
    print(f"[推理器-{worker_id}] 啟動")
    # YOLO 初始化也在執行緒避免阻塞
    model = await asyncio.to_thread(YOLO, args.tflite_model, task="detect")
    while True:
        index, frame = await frame_queue.get()
        if index is None and frame is None:
            print(f"[推理器-{worker_id}] 結束")
            break
        # 推理與繪圖皆在執行緒
        result = (await asyncio.to_thread(model.predict, frame, verbose=False))[0]
        img = await asyncio.to_thread(result.plot)
        await asyncio.to_thread(cv2.imshow, f"worker-{worker_id}", img)
        await asyncio.to_thread(cv2.waitKey, 1)

async def main():
    frame_queue = asyncio.Queue(maxsize=100)
    workers = [asyncio.create_task(inference_worker(i, frame_queue)) for i in range(args.max_workers)]
    await preprocess(frame_queue)
    await asyncio.gather(*workers)

if __name__ == "__main__":
    asyncio.run(main())
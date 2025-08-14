
import cv2
import argparse
import asyncio
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLO Asyncio Inference")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
parser.add_argument("--max_workers", type=int, default=5, help="Number of inference workers")
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

async def inference_worker(worker_id: int, frame_queue: asyncio.Queue, display_queue: asyncio.Queue):
    print(f"[推理器-{worker_id}] 啟動")
    model = await asyncio.to_thread(YOLO, args.tflite_model, task="detect")
    while True:
        index, frame = await frame_queue.get()
        if index is None and frame is None:
            print(f"[推理器-{worker_id}] 結束")
            # 通知顯示 queue 結束
            await display_queue.put((worker_id, None, None))
            break
        result = (await asyncio.to_thread(model.predict, frame, verbose=False))[0]
        # 將 result 物件直接傳給 display_queue，避免在子執行緒呼叫 plot()
        await display_queue.put((worker_id, index, result))

async def display_loop(display_queue: asyncio.Queue, num_workers: int):
    finished_workers = 0
    while finished_workers < num_workers:
        worker_id, index, result = await display_queue.get()
        if result is None:
            finished_workers += 1
            continue
        # 在主執行緒呼叫 plot()
        img = result.plot()
        cv2.imshow(f"worker-{worker_id}", img)
        cv2.waitKey(1)

async def main():
    frame_queue = asyncio.Queue(maxsize=100)
    display_queue = asyncio.Queue(maxsize=100)
    workers = [asyncio.create_task(inference_worker(i, frame_queue, display_queue)) for i in range(args.max_workers)]
    display_task = asyncio.create_task(display_loop(display_queue, args.max_workers))
    await preprocess(frame_queue)
    await asyncio.gather(*workers)
    await display_task

if __name__ == "__main__":
    asyncio.run(main())
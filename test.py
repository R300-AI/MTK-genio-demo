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

frame_queue = asyncio.Queue(maxsize=10)
result_queue = asyncio.Queue(maxsize=10)

executor = ThreadPoolExecutor(max_workers=max(1, args.max_workers + 1))

async def reader():
    loop = asyncio.get_running_loop()
    frame_count = 0
    print("[讀取器] 啟動")
    while True:
        ret, frame = await loop.run_in_executor(executor, cap.read)
        if not ret:
            print(f"[讀取器] 影片結束，總共讀取 {frame_count} 幀")
            break
        try:
            await frame_queue.put(frame)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[讀取器] 已放入第 {frame_count} 幀")
        except asyncio.QueueFull:
            print("[讀取器] frame_queue 已滿，等待中...")
            continue
    # 放入 None 給每個 predictor worker
    for _ in range(args.max_workers):
        await frame_queue.put(None)
    cap.release()
    print("[讀取器] 結束")

async def predictor():
    loop = asyncio.get_running_loop()
    proc_id = id(asyncio.current_task())
    count = 0
    print(f"[推理器-{proc_id}] 啟動")
    while True:
        frame = await frame_queue.get()
        if frame is None:
            print(f"[推理器-{proc_id}] 收到結束訊號，離開。共處理 {count} 幀")
            break
        result = await loop.run_in_executor(executor, lambda: model.predict(frame, verbose=False)[0])
        try:
            await result_queue.put((result, frame))
            count += 1
            if count % 10 == 0:
                print(f"[推理器-{proc_id}] 已推理 {count} 幀")
        except asyncio.QueueFull:
            print(f"[推理器-{proc_id}] result_queue 已滿，等待中...")
            continue
    print(f"[推理器-{proc_id}] 結束")

async def displayer():
    loop = asyncio.get_running_loop()
    count = 0
    print("[顯示器] 啟動")
    while True:
        try:
            result, frame = await result_queue.get()
        except Exception:
            print("[顯示器] result_queue 例外，重試中...")
            continue
        img = await loop.run_in_executor(executor, lambda: cv2.resize(result.plot(), (720, 480)))
        await loop.run_in_executor(executor, lambda: cv2.imshow("Ultralytics Async Inference", img))
        key = await loop.run_in_executor(executor, lambda: cv2.waitKey(1))
        count += 1
        if count % 10 == 0:
            print(f"[顯示器] 已顯示 {count} 幀")
        if key & 0xFF == ord('q'):
            print("[顯示器] 收到退出訊號 (q)")
            break
    cv2.destroyAllWindows()
    print(f"[顯示器] 結束，總共顯示 {count} 幀")

async def main():
    tasks = [asyncio.create_task(reader())]
    for _ in range(args.num_workers):
        tasks.append(asyncio.create_task(predictor()))
    tasks.append(asyncio.create_task(displayer()))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
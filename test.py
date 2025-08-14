from ultralytics import YOLO
import cv2, argparse
import asyncio

parser = argparse.ArgumentParser(description="YOLO Async Inference with asyncio")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
model = YOLO(args.tflite_model, task="detect")

frame_queue = asyncio.Queue(maxsize=100)
result_queue = asyncio.Queue(maxsize=100)

async def preprocess():
    print("[讀取器] 啟動")
    loop = asyncio.get_running_loop()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        await frame_queue.put(frame)
    cap.release()
    print("[讀取器] 結束")

async def inference():
    print(f"[推理器] 啟動")
    loop = asyncio.get_running_loop()
    while True:
        frame = await frame_queue.get()
        if frame is None:
            await result_queue.put(None)
            print(f"[推理器] 收到結束訊號，離開。")
            break
        result = model.predict(frame, verbose=False)[0]
        print(result.plot().shape)

async def main():
    tasks = [
        asyncio.create_task(preprocess()),
        asyncio.create_task(inference())
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

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

async def reader():
    print("[讀取器] 啟動")
    loop = asyncio.get_running_loop()
    while True:
        ret, frame = cap.read()
        if not ret:
            await frame_queue.put(None)
            break
        await frame_queue.put(frame)
    
    cap.release()
    print("[讀取器] 結束")

def predict(frame, model):
    result = model.predict(frame, verbose=False)[0]
    result_queue.put((frame, model, result))
    
async def predictor():
    loop = asyncio.get_running_loop()
    print(f"[推理器] 啟動")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        while True:
            frame = await frame_queue.get()
            if frame is None:
                break
            # 射後不理地送出任務
            fut = loop.run_in_executor(
                executor,
                predict_and_put,
                frame,
                model,
                result_queue
            )
            futures.append(fut)
        # 等所有任務完成
        await asyncio.gather(*futures)
        await result_queue.put(None)
    print(f"[推理器] 結束")

async def displayer():
    print("[顯示器] 啟動")
    loop = asyncio.get_running_loop()
    while True:
        result = await result_queue.get()
        if result is None:
            print(f"[顯示器] 收到結束訊號，離開。")
            break
        cv2.imshow("Ultralytics Async Inference", cv2.resize(result.plot(), (720, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[顯示器] 收到退出訊號 (q)")
            break
    cv2.destroyAllWindows()
    print(f"[顯示器] 結束")

async def main():
    tasks = [
        asyncio.create_task(reader()),
        asyncio.create_task(predictor()),
        asyncio.create_task(displayer())
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
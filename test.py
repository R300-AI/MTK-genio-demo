from ultralytics import YOLO
import cv2
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

async def async_predict(model, frame, executor):
    """異步執行 YOLO 推理"""
    print("[階段] 開始執行異步推理")
    loop = asyncio.get_event_loop()
    # 將 CPU 密集的推理任務放到線程池中執行
    result = await loop.run_in_executor(executor, lambda: model.predict(frame, verbose=False)[0])
    print("[階段] 推理結束")
    return result

async def async_frame_processing(cap, model, executor):
    """異步處理視頻幀"""
    print("[階段] 開始異步處理視頻幀")
    while True:
        # 讀取幀也可能是 I/O 操作，放到線程池中
        loop = asyncio.get_event_loop()
        print("[階段] 讀取視頻幀")
        ret, frame = await loop.run_in_executor(executor, cap.read)
        
        if not ret:
            print("[階段] 視頻結束或讀取失敗，結束處理")
            break
            
        # 異步執行推理
        result = await async_predict(model, frame, executor)
        
        # 顯示結果
        print("[階段] 顯示推理結果")
        display_frame = cv2.resize(result.plot(), (720, 480))
        cv2.imshow("Ultralytics Async Inference", display_frame)
        
        # 檢查是否按下 ESC 鍵退出
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            print("[階段] 偵測到 ESC 鍵，結束處理")
            break
            
        # 讓出控制權給事件循環
        await asyncio.sleep(0.001)

async def main():
    print("[階段] 解析命令列參數")
    parser = argparse.ArgumentParser(description="YOLO Async Inference with asyncio")
    parser.add_argument("--video_path", type=str, default="./data/video.mp4")
    parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
    parser.add_argument("--max_workers", type=int, default=4, help="線程池最大工作線程數")
    args = parser.parse_args()

    print("[階段] 初始化視頻捕獲和模型")
    cap = cv2.VideoCapture(args.video_path)
    model = YOLO(args.tflite_model, task="detect")
    
    print("[階段] 創建線程池執行器")
    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    
    try:
        print("[階段] 執行異步幀處理")
        await async_frame_processing(cap, model, executor)
    finally:
        print("[階段] 清理資源")
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True)

if __name__ == "__main__":
    print("[階段] 啟動主程序")
    asyncio.run(main())
"""
from ultralytics import YOLO
import cv2, argparse

parser = argparse.ArgumentParser(description="YOLO Async Inference with Subprocess")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
model = YOLO(args.tflite_model, task="detect")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = model.predict(frame, verbose=False)[0]
    cv2.imshow("Ultralytics Async Inference", cv2.resize(result.plot(), (720, 480)))

cap.release()
"""

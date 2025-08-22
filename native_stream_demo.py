from ultralytics import YOLO
import cv2
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time

parser = argparse.ArgumentParser(description="YOLO Streaming Inference")
parser.add_argument("--video_path", type=str, default="./data/video.mp4")
parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
model = YOLO(args.tflite_model, task="detect")
output_queue = Queue(maxsize=10)
executor = ThreadPoolExecutor(max_workers=2)  # 用於異步推論

def predict_frame(frame):
    """異步推論函數"""
    result = model.predict(frame, verbose=False, stream=False)[0]  # 不用stream
    return result

def frame_reader():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 提交推論任務到線程池，返回Future
        future = executor.submit(predict_frame, frame)
        try:
            print('[FUTURE]', frame.shape)
            output_queue.put(future, timeout=0.1)
        except:
            continue  # 如果queue滿了，跳過這一frame
    output_queue.put(None)

def frame_displayer():
    while True:
        future = output_queue.get(timeout=10.0)
        if future is None:
            break
        # 等待異步推論完成並取得結果
        result = future.result()  # 這裡會等待推論完成
        print('[DISPLAY]', result.orig_img.shape)
        processed_frame = result.plot()
        cv2.imshow("YOLO Streaming Inference", cv2.resize(processed_frame, (720, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

reader_thread = threading.Thread(target=frame_reader)
displayer_thread = threading.Thread(target=frame_displayer)

reader_thread.start()
displayer_thread.start()

reader_thread.join()
displayer_thread.join()

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)  # 關閉線程池
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
from ultralytics import YOLO

model = YOLO("./models/yolov8n_float32.tflite")
results = model(["./data/bus.jpg"])
results[0].show()

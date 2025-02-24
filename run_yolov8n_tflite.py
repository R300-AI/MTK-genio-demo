from utils.ultralytics.tflite import preprocess, postprocess, visualizer
import tflite_runtime.interpreter as tflite_runtime
import matplotlib.pyplot as plt
import numpy as np
import cv2, pickle, time

with open ('coco_labels.txt', 'rb') as f:
    classes = pickle.load(f)
    nc, labels = len(classes), {i: cls for i, cls in enumerate(classes)}
    
#require python3.7
img = cv2.imread('./bus.jpg')

interpreter = tflite_runtime.Interpreter(model_path='./models/yolov8n_float32.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
imgsz = input_details[0]['shape'][1:3]   # (640, 640)
interpreter.allocate_tensors()

# Step1. Preprocess
inputs = preprocess([img], imgsz = imgsz, dtype = np.float32)

# Step2. Run inference with Tenrflow Lite
for _ in range(10):
    t = time.time()
    interpreter.set_tensor(input_details[0]['index'], inputs)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    print(time.time() - t)

# Step3. Postprocess
results = postprocess(preds, imgsz, nc = nc, conf_thres=0.25, iou_thres=0.7, agnostic=False, max_det=300)

# Step4. Visualization
plt.imshow(visualizer(img.copy(), results[0].copy(), labels)[:, :, ::-1])
plt.show()

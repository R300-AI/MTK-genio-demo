from utils.ultralytics.tflite import preprocess, postprocess, visualizer
from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
from utils.neuronpilot import neuronrt
import matplotlib.pyplot as plt
import cv2, subprocess, pickle, time
import numpy as np

##############[ Configuration ]##############


with open ('./resources/coco_labels.txt', 'rb') as f:
    classes = pickle.load(f)
    nc, labels = len(classes), {i: cls for i, cls in enumerate(classes)}

output_handlers_with_shape = {'./bin/output0.bin': (1, 1, 84, 8400)}   # you can use the 'sudo neuronrt -a yolov8n_float32.dla -d' cammand to check the number of output handlers needed.
image_path = './resources/bus.jpg'
model_path = "./models/yolov8n_float32.dla"
imgsz = (640, 640)

# Step1. Get RGB image via OpenCV and Preprocess
img = cv2.imread(image_path)[..., :: -1]
inputs = preprocess([img], imgsz = imgsz, dtype = np.float32)

# Step2. Convert the RGB image into binary format
bin_path = convert_to_binary(image_data = inputs, handeler = 0)

# Step3. Run inference with NeuronRT and Evaluate the Speed
# Specify the boost value for Quality of Service. Range is 0(fastest) to 100(highest accuracy).
t = time.time()
neuronrt.predict(input_handlers = [f"./bin/{bin_path}"], 
                 output_handlers  = list(output_handlers_with_shape.keys()), 
                 compiled_dla_model = model_path, 
                 boost_value = 100
                 )
print(f'Inference Speed: {(time.time() - t) * 1000} ms')

# Step4. Get the binary output and Postprocess
preds = conert_to_numpy(output_handlers_with_shape, dtype = np.float32)[0]
results = postprocess(preds, imgsz, nc = nc, conf_thres=0.25, iou_thres=0.7, agnostic=False, max_det=300)

# Step5. Visualization
plt.imshow(visualizer(img.copy(), results[0].copy(), labels))
plt.show()

# Importing DLA execution lbraries for YOLO Object Detection.
#
# For more details on these lbraries, refer to the following resources:
# - ultralytics.yolo: Utilities for preprocessing, postprocessing, and visualizing YOLO outputs.
# - neuronpilot.data: Data conversion between binary or numpy formats for model input.
# - neuronpilot: Runtime for executing models on neuron hardware.
from utils.ultralytics.yolo import preprocess, postprocess, visualizer
from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
from utils.neuronpilot import neuronrt

import matplotlib.pyplot as plt
import cv2, subprocess, pickle, time
import numpy as np

# Model configuration settings.
#
# Adjust these parameters as required:
# - image_path: Path to the input image file for detection.
# - model_path: Path to the compiled YOLO model file.
# - imgsz: Size of the input image for the model.
# - output_handlers_with_shape: Defining output handlers and their shapes.   (you can use the 'sudo neuronrt -a yolov8n_float32.dla -d' to check it out.)
# - classes: List of object detection classes from the trained dataset.
# - labels: Mapping of class indices to their respective names.
# - nc: Total number of object detection classes.
image_path = './data/bus.jpg'
model_path = "./models/yolov8n_float32.dla"
imgsz = (640, 640)
output_handlers_with_shape = {'./bin/output0.bin': (1, 1, 84, 8400)}

with open ('./data/coco_labels.txt', 'rb') as f:
    classes = pickle.load(f)
    nc, labels = len(classes), {i: cls for i, cls in enumerate(classes)}


# This script demonstrates the end-to-end process of running a YOLOv8n model using NeuronRT.
# It includes the following steps:
# 1. Read and preprocess an input image using OpenCV.
# 2. Convert the preprocessed image into a binary format suitable for model input.
# 3. Perform inference using the NeuronRT runtime and measure the inference speed.
# 4. Convert the binary output back to a numpy array and postprocess the results to obtain final detections.
# 5. Visualize the detection results on the original image using matplotlib.

# Step1. Get RGB image via OpenCV and Preprocess
img = cv2.imread(image_path)[..., :: -1]
inputs = preprocess([img], imgsz = imgsz, dtype = np.float32)

# Step2. Convert the RGB image into binary format
bin_path = convert_to_binary(image_data = inputs, handeler = 0)

# Step 3. Run inference with NeuronRT and evaluate the speed
t = time.time()
neuronrt.predict(input_handlers = [f"./bin/{bin_path}"], 
                 output_handlers  = list(output_handlers_with_shape.keys()), 
                 compiled_dla_model = model_path, 
                 boost_value = 100   # Quality of Service: range from 0 (fastest) to 100 (highest accuracy)
                 )
print(f'Inference Speed: {(time.time() - t) * 1000} ms')

# Step4. Get the binary output and Postprocess
preds = conert_to_numpy(output_handlers_with_shape, dtype = np.float32)[0]
results = postprocess(preds, imgsz, nc = nc, conf_thres=0.25, iou_thres=0.7, agnostic=False, max_det=300)

# Step5. Visualization
plt.imshow(visualizer(img.copy(), results[0].copy(), labels))
plt.show()

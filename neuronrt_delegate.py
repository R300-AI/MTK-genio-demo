from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
from utils.tools import Neuronpilot_WebAPI
from utils.neuronpilot import neuronrt
import argparse, time, warnings
import numpy as np
import tensorflow as tf

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--tflite_model", type=str, help="Path to .tflite")
parser.add_argument("-t", "--iteration", default=10, type=int, help="Test How Many Times?")
args = parser.parse_args()

#dla_path = Neuronpilot_WebAPI(
#    tflite_path = args.tflite_model, output_folder = './models', 
#    url = 'https://app-aihub-neuronpilot.azurewebsites.net/')
#print(f"Converted file saved at: {dla_path}")
dla_path = './models/yolov8n_float32.dla'

interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
output_handlers_with_shape = {f'./bin/output{i}.bin': tuple([1] + output['shape'].tolist()) for i, output in enumerate(output_details)}
print(output_handlers_with_shape)

inputs = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])

t = time.time()
for _ in range(args.iteration):
  bin_path = convert_to_binary(image_data = inputs, handeler = 0)
  neuronrt.predict(input_handlers = [f"./bin/{bin_path}"], 
                   output_handlers  = list(output_handlers_with_shape.keys()), 
                   compiled_dla_model = dla_path, 
  )
  preds = conert_to_numpy(output_handlers_with_shape, dtype = np.float32)[0]
print(f'Inference Speed: {(time.time() - t) * 1000 / args.iteration} ms')
  



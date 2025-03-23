from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
from utils.tools import Neuronpilot_WebAPI
from utils.neuronpilot import neuronrt
import argparse, time, warnings, shutil, os
import numpy as np
import tensorflow as tf

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--tflite_model", type=str, help="Path to .tflite")
parser.add_argument("-t", "--iteration", default=10, type=int, help="Test How Many Times?")
args = parser.parse_args()

# Compile TFLite model to DLA format via NeuronPilot WebAPI if DLA not exist.
if not os.path.exists(args.tflite_path.replace('.tflite', '.dla')):
  dla_path = Neuronpilot_WebAPI(
      tflite_path = args.tflite_model, output_folder = './models', 
      url = 'https://app-aihub-neuronpilot-v2.azurewebsites.net/')
  print(f"Converted file saved to: {dla_path}")
else:
    dla_path = args.tflite_path.replace('.tflite', '.dla')
    print(f"Load existed dla model: {self.dla_path}")
if os.path.exists('./bin'):
  shutil.rmtree('./bin')
  os.mkdir('./bin')

# Initialize Interpreter
neuronrt.load(dla_path)
interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
output_handlers_with_shape = {f'./bin/output{i}.bin': tuple([1] + output['shape'].tolist()) for i, output in enumerate(output_details)}

# Forward Propagation
inputs = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])

t1 = time.time()
bin_path = convert_to_binary(image_data = inputs, handeler = 0)
t2 = time.time()
for _ in range(args.iteration):
  neuronrt.predict(input_handlers = [bin_path], 
                   output_handlers  = list(output_handlers_with_shape.keys()), 
                   compiled_dla_model = dla_path, 
  )
t3 = time.time()
preds = conert_to_numpy(output_handlers_with_shape, dtype = np.float32)[0]

print(f'Set tensor speed: {(t2 - t1) * 1000} ms')
print(f'Inference speed: {(t3 - t2) * 1000 / args.iteration} ms')
print(f'Get tensor speed: {(time.time() - t3) * 1000} ms')


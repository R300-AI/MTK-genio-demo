from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
from utils.tools import Neuronpilot_WebAPI
from utils.neuronpilot import neuronrt
import numpy as np
import tensorflow as tf
import os, shutil

class Interpreter():
    def __init__(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.tflite_path = model_path
        self.output_handlers_with_shape = {f'./bin/output{i}.bin': tuple([1] + output['shape'].tolist()) for i, output in enumerate(self.output_details)}
        
    def allocate_tensors(self):
        if not os.path.exists(self.tflite_path.replace('.tflite', '.dla')):
           self.dla_path = Neuronpilot_WebAPI(
               tflite_path = self.tflite_path, output_folder = './models', 
               url = 'https://app-aihub-neuronpilot.azurewebsites.net/'
           )
           print(f"Converted file saved to: {self.dla_path}")
        else:
           self.dla_path = self.tflite_path.replace('.tflite', '.dla')
           print(f"Load existed dla model: {self.dla_path}")
        
        if os.path.exists('./bin'):
           shutil.rmtree('./bin')
           os.mkdir('./bin')

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details
   
    def set_tensor(self, _, inputs):
        inputs = np.array(inputs).astype(self.input_details[0]['dtype'])
        self.bin_path = convert_to_binary(image_data = inputs, handeler = 0)

    def invoke(self):
        neuronrt.predict(input_handlers = [self.bin_path], 
            output_handlers  = list(self.output_handlers_with_shape.keys()), 
            compiled_dla_model = self.dla_path, 
        )
        
    def get_tensor(self, _):
        return conert_to_numpy(self.output_handlers_with_shape, dtype = np.float32)[0]

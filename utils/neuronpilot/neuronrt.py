from utils.neuronpilot.data import convert_to_binary, conert_to_numpy
import numpy as np
import tensorflow as tf
import os, shutil, subprocess

class Interpreter():
    def __init__(self, tflite_path, dla_path, device):
        self.tflite_path = tflite_path
        self.dla_path = dla_path
        self.device = device
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        
    def allocate_tensors(self):
        self.seed = int(np.random.rand() * 10e12)
        
        print(f"Load DLA model: {self.dla_path}")
        commands = ["sudo", "neuronrt", "-a", self.dla_path, "-d"]
        results = subprocess.run(commands, capture_output=True, text=True)
        
        if not os.path.exists('./bin'):
            os.makedirs('./bin')
        if os.path.exists(f'./bin/{self.seed}'):
           shutil.rmtree(f'./bin/{self.seed}')
        os.mkdir(f'./bin/{self.seed}') 
        self.input_handlers = [f'./bin/{self.seed}/input{i}.bin' for i, input in enumerate(self.input_details)]
        self.output_handlers_with_shape = {f'./bin/{self.seed}/output{i}.bin': tuple([1] + output['shape'].tolist()) for i, output in enumerate(self.output_details)}

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details
   
    def set_tensor(self, _, inputs):
        for input, binary_path in zip(inputs, self.input_handlers):
            input_data = np.array([input]).astype(self.input_details[0]['dtype'])
            self.bin_path = convert_to_binary(input_data, binary_path)

    def invoke(self):
        commands = ["sudo", "neuronrt",  
                "-m",  "hw",  
                "-a",  self.dla_path,
                "-c",  "1",         # Repeat the inference <num> times. It can be used for profiling.
                "-b",  "100",     
                "-i",  ' -i '.join(self.input_handlers),  
                "-o",  ' -o '.join(list(self.output_handlers_with_shape.keys()))]
        p = subprocess.Popen(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()
        
    def get_tensor(self, _):
        return conert_to_numpy(self.output_handlers_with_shape, dtype = np.float32)[0]
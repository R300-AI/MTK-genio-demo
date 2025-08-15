from utils.neuronpilot.data import convert_to_binary, convert_to_numpy
import numpy as np
import tensorflow as tf
import os, shutil, subprocess, uuid, queue

class Interpreter():
    def __init__(self, tflite_path, dla_path, device, bin_dir = './bin'):
        self.tflite_path = tflite_path
        self.dla_path = dla_path
        self.device = device
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.input_tensors = queue.Queue()
        self.output_tensors = queue.Queue()
        self.bin_dir = f'./bin'

    def allocate_tensors(self):
        commands = ["sudo", "neuronrt", "-a", self.dla_path, "-d"]
        results = subprocess.run(commands, capture_output=True, text=True)

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details
   
    def set_tensor(self, _, inputs):
        handler = str(uuid.uuid4())
        input_handlers = [f'{self.bin_dir}/input{handler}_{i}.bin' for i, input in enumerate(self.input_details)]
        output_handlers = {f'{self.bin_dir}/output{handler}_{i}.bin': tuple([1] + output['shape'].tolist()) for i, output in enumerate(self.output_details)}

        for input, binary_path in zip(inputs, input_handlers):
            input_data = np.array([input]).astype(self.input_details[0]['dtype'])
            convert_to_binary(input_data, binary_path)
        self.input_tensors.put((input_handlers, output_handlers))

    def invoke(self):
        input_handlers, output_handlers = self.input_tensors.get()
        commands = ["sudo", "neuronrt",  
                "-m",  "hw",  
                "-a",  self.dla_path,
                "-c",  "1",         # Repeat the inference <num> times. It can be used for profiling.
                "-b",  "100",     
                "-i",  ' -i '.join(input_handlers),  
                "-o",  ' -o '.join(list(output_handlers.keys()))]
        p = subprocess.Popen(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()
        self.output_tensors.put(output_handlers)
        
    def get_tensor(self, _):
        output_handlers = self.output_tensors.get()
        return convert_to_numpy(output_handlers, dtype = np.float32)[0]

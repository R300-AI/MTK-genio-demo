from utils.neuronpilot.data import convert_to_binary, convert_to_numpy
import numpy as np
import tensorflow as tf
import os, shutil, subprocess, uuid, queue

class Interpreter():
    def __init__(self, tflite_path, dla_path, device):
        self.tflite_path = tflite_path
        self.dla_path = dla_path
        self.device = device
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.input_handlers = queue.Queue()
        self.output_handlers = queue.Queue()
        
    def allocate_tensors(self):
        self.current_seed = str(uuid.uuid4())
        bin_dir = f'./bin'
        
        print(f"Load DLA model: {self.dla_path}")
        commands = ["sudo", "neuronrt", "-a", self.dla_path, "-d"]
        results = subprocess.run(commands, capture_output=True, text=True)
        if os.path.exists(bin_dir):
           shutil.rmtree(bin_dir)
        os.mkdir(bin_dir)

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details
   
    def set_tensor(self, _, inputs):
        handler = str(uuid.uuid4())
        input_handlers = [f'./bin/input{handler}_{i}.bin' for i, input in enumerate(self.input_details)]
        output_handlers = {f'./bin/output{handler}_{i}.bin': tuple([1] + output['shape'].tolist()) for i, output in enumerate(self.output_details)}

        for input, binary_path in zip(inputs, input_handlers):
            input_data = np.array([input]).astype(self.input_details[0]['dtype'])
            convert_to_binary(input_data, binary_path)
        self.input_handlers.put((handler, input_handlers, output_handlers))

    def invoke(self):
        handler, input_handlers, output_handlers = self.input_handlers.get()
        commands = ["sudo", "neuronrt",  
                "-m",  "hw",  
                "-a",  self.dla_path,
                "-c",  "1",         # Repeat the inference <num> times. It can be used for profiling.
                "-b",  "100",     
                "-i",  ' -i '.join(input_handlers),  
                "-o",  ' -o '.join(list(output_handlers.keys()))]
        p = subprocess.Popen(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.output_handlers.put((handler, output_handlers, p))
        
    def get_tensor(self, _):
        handler, output_handlers, p = self.output_handlers.get()
        p.wait()
        return convert_to_numpy(output_handlers, dtype = np.float32)[0]
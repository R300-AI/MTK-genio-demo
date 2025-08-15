from utils.neuronpilot.data import convert_to_binary, convert_to_numpy
import numpy as np
import tensorflow as tf
import os, shutil, subprocess, uuid, queue

class Interpreter():
    _dla_loaded = False

    def __init__(self, tflite_path, dla_path, device, admin_password='mediatekg510'):
        self.tflite_path = tflite_path
        self.dla_path = dla_path
        self.device = device
        self.admin_password = admin_password
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.input_tensors = queue.Queue()
        self.output_tensors = queue.Queue()
        
    def allocate_tensors(self):
        self.bin_dir = f'./bin'
        try:
            # 使用 Popen 來更好地控制 stdin
            process = subprocess.Popen(
                ['sudo', '-S', 'neuronrt', '-a', self.dla_path, '-d'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 將密碼寫入 stdin
            stdout, stderr = process.communicate(input=f"{self.admin_password}\n")
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, 'neuronrt', stderr)
                
            self._dla_loaded = True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load DLA model: {e.stderr}")

        if os.path.exists(self.bin_dir):
           shutil.rmtree(self.bin_dir)
        os.mkdir(self.bin_dir)

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
        # 修正命令構建問題
        commands = [
            "sudo", "neuronrt",  
            "-m",  "hw",  
            "-a",  self.dla_path,
            "-c",  "1",         
            "-b",  "100"
        ]
        
        # 添加輸入檔案
        for input_file in input_handlers:
            commands.extend(["-i", input_file])
            
        # 添加輸出檔案
        for output_file in output_handlers.keys():
            commands.extend(["-o", output_file])
        
        # 使用密碼執行推理
        try:
            p = subprocess.run(
                command_str = ' '.join(commands),
                shell=True,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                check=True
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"Inference failed: {e}")
            
        self.output_tensors.put(output_handlers)
        for input_file in input_handlers:
            try:
                os.remove(input_file)
            except OSError:
                pass
        
    def get_tensor(self, _):
        output_handlers = self.output_tensors.get()
        for output_file in output_handlers:
            try:
                os.remove(output_file)
            except OSError:
                pass
        return convert_to_numpy(output_handlers, dtype = np.float32)[0]

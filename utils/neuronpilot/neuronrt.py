from utils.neuronpilot.data import convert_to_binary, convert_to_numpy
import numpy as np
import tensorflow as tf
import os, shutil, subprocess, uuid

class Interpreter():
    """NeuronRT Interpreter - 專為單序列推論優化"""
    def __init__(self, tflite_path, dla_path, device):
        self.tflite_path = tflite_path
        self.dla_path = dla_path
        self.device = device
        
        # 為每個 interpreter 實例創建唯一的 bin 目錄
        self.id = str(uuid.uuid4())
        self.bin_dir = f'./bin/{self.id}'
        os.makedirs(self.bin_dir, exist_ok=True)

        # 使用 TFLite 解釋器獲取模型資訊
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        
        # 單序列使用固定的檔案處理器
        self.input_handlers = [f'{self.bin_dir}/input_{i}.bin' for i in range(len(self.input_details))]
        self.output_handlers = {
            f'{self.bin_dir}/output_{i}.bin': tuple([1] + output['shape'].tolist()) 
            for i, output in enumerate(self.output_details)
        }

    def allocate_tensors(self):
        """分配張量資源"""
        commands = ["sudo", "neuronrt", "-a", self.dla_path, "-d"]
        results = subprocess.run(commands, capture_output=True, text=True)

    def get_input_details(self):
        """獲取輸入詳細資訊"""
        return self.input_details

    def get_output_details(self):
        """獲取輸出詳細資訊"""
        return self.output_details
   
    def set_tensor(self, _, inputs):
        """設定輸入張量 - 簡化版本直接處理輸入"""
        # 直接將輸入數據寫入預定義的檔案
        for input_data, binary_path in zip(inputs, self.input_handlers):
            input_array = np.array([input_data]).astype(self.input_details[0]['dtype'])
            convert_to_binary(input_array, binary_path)

    def invoke(self):
        """執行推論 - 簡化版本"""
        commands = ["sudo", "neuronrt",  
                "-m",  "hw",  
                "-a",  self.dla_path,
                "-c",  "1",         # 單次推論
                "-b",  "100",     
                "-i",  ' -i '.join(self.input_handlers),  
                "-o",  ' -o '.join(list(self.output_handlers.keys()))]
        
        # 同步執行，等待完成
        p = subprocess.Popen(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()
        
    def get_tensor(self, _):
        """獲取輸出張量 - 簡化版本直接返回結果"""
        return convert_to_numpy(self.output_handlers, dtype=np.float32)[0]

from utils.neuronpilot.data import convert_to_binary, convert_to_numpy
import numpy as np
import os, shutil, subprocess, uuid
import tensorflow as tf
import gc

class Interpreter():
    """NeuronRT Interpreter - 專為單序列推論優化"""
    def __init__(self, dla_path):
        self.dla_path = dla_path

        if not dla_path.endswith('.dla'):
            raise ValueError("dla_path格式錯誤，應為 <model_name>.tflite.<device>.dla")
            
        parts = dla_path.split('.tflite.')
        if len(parts) != 2 or not parts[1].endswith('.dla'):
            raise ValueError("dla_path格式錯誤，應為 <model_name>.tflite.<device>.dla")
        
        # 取得 device 並轉換格式
        device_name = parts[1][:-4]  # 去掉.dla
        if device_name.startswith('mdla') and '.' not in device_name:
            self.device = device_name + '.0'  # mdla3 -> mdla3.0, mdla2 -> mdla2.0
        else:
            self.device = device_name  # vpu 等其他 device 保持不變

        self.tflite_path = parts[0] + '.tflite'

        if not os.path.exists(self.dla_path):
            raise FileNotFoundError(f"找不到DLA模型檔案: {self.dla_path}")
        if not os.path.exists(self.tflite_path):
            raise FileNotFoundError(f"找不到{self.dla_path}對應的TFLite檔案: {self.tflite_path}")

        # 為每個 interpreter 實例創建唯一的 bin 目錄
        self.id = str(uuid.uuid4())
        self.bin_dir = f'./bin/{self.id}'
        os.makedirs(self.bin_dir, exist_ok=True)

        # 輕量取得 TFLite 模型的 input/output details (不載入模型權重)
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        del interpreter  # 立即釋放
        gc.collect()     # 強制垃圾回收，釋放記憶體

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

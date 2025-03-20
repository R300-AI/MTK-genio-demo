# Deploy Computer Vision Applications Using Ultralytics YOLO

## Download and Config Ultralytics for ArmNN and NeuronRT
第一步、透過script下載並設定ultralytics的source code
```bash
$ git clone https://github.com/R300-AI/MTK-genio-demo.git
$ cd MTK-genio-demo
$ bash ./build.sh
```

第二步、將`./ultralytics/nn/autobackend.py` 419行初始化TFLite解釋器的位置註解.

  ```python
  # interpreter = Interpreter(model_path=w)  # load TFLite model
  ```

第三步、將註解的部分替換為以下內容

  ```python
  armnn_delegate = load_delegate(
      library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
      options={"backends": 'GpuAcc', "logging-severity":"info"}
  )
  interpreter = Interpreter(
      model_path=w, 
      experimental_delegates=[armnn_delegate]
  )    
  ```
  or
  ```python
  from utils.neuronpilot import runtime

  interpreter = runtime.Interpreter(model_path=w)
  ```
  if you want to use neuronrt

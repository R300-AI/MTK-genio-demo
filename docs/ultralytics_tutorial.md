# Deploy Computer Vision Applications Using Ultralytics YOLO

## Installation
第一步、透過script下載並設定ultralytics的source code
```bash
$ git clone https://github.com/R300-AI/MTK-genio-demo.git
$ cd MTK-genio-demo
$ bash ./build.sh
```

第二部、將`./ultralytics/nn/autobackend.py` 大約409 ~ 415行初始化TFLite解釋器的位置註解.

  ```python
  $ # delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
  $ #    platform.system()
  $ # ]
  $ # interpreter = Interpreter(
  $ #    model_path=w,
  $ #    experimental_delegates=[load_delegate(delegate, options={"device": device})],
  $ # )
  ```

第三步、將註解的部分替換為以下內容

  ```python
  armnn_delegate = load_delegate(
      library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
      options={"backends": 'GpuAcc', "logging-severity":"info"}
  )
  interpreter = Interpreter(
      model_path=args.tflite_model, 
      experimental_delegates=[armnn_delegate]
  )    
  ```

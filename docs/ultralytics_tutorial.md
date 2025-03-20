# Deploy Language Process Applications Using Ollama

## Installation
第一步、透過script下載並設定ultralytics的source code
```bash
$ git clone https://github.com/R300-AI/MTK-genio-demo.git
$ cd MTK-genio-demo
$ bash ./build.sh
```

./ultralytics/nn/autobackend.py
  ```bash
  # delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
  #    platform.system()
  #]
  #interpreter = Interpreter(
  #    model_path=w,
  #    experimental_delegates=[load_delegate(delegate, options={"device": device})],
  #)
  ```
  ```bash
  armnn_delegate = load_delegate(
      library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
      options={"backends": 'GpuAcc', "logging-severity":"info"}
  )
  interpreter = Interpreter(
      model_path=args.tflite_model, 
      experimental_delegates=[armnn_delegate]
  )    
  ```

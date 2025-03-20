# Deploy Computer Vision Applications Using Ultralytics YOLO

## Installation
第一步、透過script下載並設定ultralytics的source code
```bash
$ git clone https://github.com/R300-AI/MTK-genio-demo.git
$ cd MTK-genio-demo
$ bash ./build.sh
```

第二部、將`./ultralytics/nn/autobackend.py`大約409 ~ 415行及717 ~ 721行的位置註解
  **Section A**
  ```bash
  $ # delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
  $ #    platform.system()
  $ # ]
  $ # interpreter = Interpreter(
  $ #    model_path=w,
  $ #    experimental_delegates=[load_delegate(delegate, options={"device": device})],
  $ # )
  ```
  **Section B**
  ```bash
  $ # self.interpreter.set_tensor(details["index"], im)
  $ # self.interpreter.invoke()
  $ # y = []
  $ # for output in self.output_details:
  $ #     x = self.interpreter.get_tensor(output["index"])
  ```

第三步、將上述替換為ArmNN及NeuronRT Delegation


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

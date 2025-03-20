# Deploy Computer Vision Applications Using Ultralytics YOLO

## Configure a Custom Ultralytics for ArmNN and NeuronRT

### Step 1: Download and Set Up Ultralytics Source Code
First, download and set up the Ultralytics source code using the provided script.
```bash
$ git clone https://github.com/R300-AI/MTK-genio-demo.git
$ cd MTK-genio-demo
$ bash ./build.sh
```

### Step 2: Modify the TFLite Interpreter Initialization
  Next, comment out the initialization of the TFLite interpreter at line 419 in　`./ultralytics/nn/autobackend.py`.

  ```python
  if edgetpu:
      device = device[3:] if str(device).startswith("tpu") else ":0"
      LOGGER.info(f"Loading {w} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
      delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
          platform.system()
      ]
      interpreter = Interpreter(
          model_path=w,
          experimental_delegates=[load_delegate(delegate, options={"device": device})],
      )
      device = "cpu"
  else:
      LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
      # interpreter = Interpreter(model_path=w)
  interpreter.allocate_tensors()  
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details() 
  ```

### Step 3: Replace the Commented Code with the Following

  Replace the commented code with the following content to use the ArmNN delegate:

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
  Alternatively, if you want to use NeuronRT, replace the commented code with the following:
  ```python
  from utils.neuronpilot import runtime

  interpreter = runtime.Interpreter(model_path=w)
  ```
### Additional Information
  * Ensure that the paths specified in the code (e.g., /home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so) are correct and accessible on your system.
  * The `options` dictionary in the `load_delegate` function allows you to specify various backend options and logging settings. Adjust these settings as needed for your specific use case.
  * The `runtime.Interpreter` class from `utils.neuronpilot` is used to initialize the NeuronRT interpreter. Make sure that the `utils.neuronpilot` module is correctly installed and accessible in your environment.

By following these steps, you will be able to configure Ultralytics YOLO to work with ArmNN and NeuronRT for deploying computer vision applications on your target hardware.

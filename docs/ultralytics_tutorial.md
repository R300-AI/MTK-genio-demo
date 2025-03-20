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
#### Additional Information
  * Ensure that the ArmNN paths specified in the code (e.g., /home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so) match the location where you have installed them.
  * You need to specify the ArmNN backend options here by setting the `backends` in `options` dictionary.
  * The `runtime.Interpreter` class mimics the functionality of the tflite runtime functions. If any modifications are needed, you can refer to the `utils.neuronpilot.runtime` file.

  By following these steps, you will be able to configure Ultralytics YOLO to work with ArmNN and NeuronRT for deploying your computer vision applications.

### Step 4: Run the Inference
  Execute the following command to run the inference using the Ultralytics framework:

  ```bash
  python ultralytics_demo.py
  ```
  This step utilizes the Ultralytics framework's execution tool to perform inference. For more information on why to use Ultralytics YOLO for inference, refer to the [Ultralytics documentation](https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference).
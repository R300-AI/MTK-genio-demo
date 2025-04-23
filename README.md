# The Best Implementation for Genio EVKs


## Requirements
* A **Genio 510/700/1200 Evaluation Kit**.
* Follow the Developer's [Document](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk.html) to build the **Ubuntu** and **Packages** to your Genio board.


## How to Use This?

<div align="center">
<img src="https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/images/chipset.png" width=360"/>
</div>

  Genio provides acceleration options for TLite models using Mali GPU and MDLA, while other operations are executed on the Cortex-A CPU. If you are interested in porting your TLite model to these chips, you can follow the instructions below to download this example repository to your Genio board and set up the necessary environment (Python=`3.12`).
  ```bash
  $ git clone https://github.com/R300-AI/MTK-genio-demo.git
  $ cd MTK-genio-demo
  $ pip install -r requirements.txt
  ```

You can use the following tools to test the computation speed of your model.

### ArmNN Benchmarks
  **ArmNN** is a neural network inference library optimized for Arm CPUs and GPUs. The file `./models/yolov8n_float32.tflite` is provided as an example and can be replaced with your own custom TFLite model.
  ```bash
  $ python armnn_benchmark.py --tflite_model ./models/yolov8n_float32.tflite --device GpuAcc --iteration 10
  ```
  > `--device`<br>
  > | Option   | Description                                   |
  > |----------|-----------------------------------------------|
  > | `CpuAcc` | Optimizes TFLite inference for Cortex-A CPUs  |
  > | `GpuAcc` | Accelerating TFLite inference for Mali-G GPUs |
  

### NeuronRT Benchmarks
  **NeuronRT** is a runtime library designed for NPU inference. It automatically compiles TFLite models into the DLA format using the [NeuronPilot Online](https://app-aihub-neuronpilot.azurewebsites.net/) API, and saves them to the `./models` directory. Please note that this process requires an active internet connection.
  ```bash
  $ python neuronrt_benchmark.py --tflite_model ./models/yolov8n_float32.tflite --device mdla3.0 --iteration 10
  ```
  > `--device`:
  > | Option    | Description                          |
  > |-----------|--------------------------------------|
  > | `mdla3.0` | Accelerating TFLite inference DLA, supported on G510/700 only |
  > | `mdla2.0` | Accelerating TFLite inference DLA, supported on 1200 only     |
  > | `vpu`     |Accelerating TFLite inference VPU                             |


## Others 

Through the following Python examples, you will gain a comprehensive understanding of the fundamentals involved in deploying AI models on these chips, including application deployment, accelerated model inference, and data streaming. This will help you quickly grasp the full potential of Genio chips. It is recommended to start exploring features similar to your use case.

### Tutorial List
* **[Deploy Computer Vision Applications Using Ultralytics YOLO](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/ultralytics_tutorial.md)**
* **[Deploy Language Process Applications Using Ollama and OpenAI Library](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/ollama_tutorial.md)**
* **[Optimizing TFLite Inference Performance with Asynchronous Streaming](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/nnstreamer_tutorial.md)**
* **[How to Manually Compile a TFLite Model into DLA Format? (TODO)]()**
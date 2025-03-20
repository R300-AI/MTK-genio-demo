# The Best Implementation for Genio EVKs

<div align="center">
<img src="https://github.com/R300-AI/ITRI-AI-Hub/blob/main/docs/assets/images/pages/genio_510_demonstration_workflow.png" width="780"/>
</div>

## Requirements
* A **Workstation** and **Genio Evaluation Kit**.
* Follow the Developer's [Document](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk.html) to build the **Ubuntu** and **Packages** to your Genio board.


## How to Use This?
  Genio provides acceleration options for TLite models using Mali GPU and MDLA, while other operations are executed on the Cortex-A CPU.

  **NeuronRT** is a runtime library for neural network inference, and **ArmNN** is a neural network inference library optimized for Arm CPUs and GPUs. If you are interested in porting your TLite model to these chips, you can follow the instructions below to download this example repository to your Genio board and set up the necessary environment (Python=`3.12`).
  ```bash
  $ git clone https://github.com/R300-AI/MTK-genio-demo.git
  $ cd MTK-genio-demo
  $ pip install -r requirements.txt
  ```

### NeuronRT Benchmarks
  You can use the following command to preliminarily test the computation speed of your model on MDLA. (**Note**: Internet access may be required to compile the tflite model into dla format)
  ```bash
  $ python neuronrt_benchmark.py --tflite ./models/yolov8n_float32.tflite --iteration 10
  ```
### ArmNN Benchmarks
  You can use the following command to test the computation speed of your model on Cortex-A(device=`CpuAcc`) or Mali-G(device=`GpuAcc`).
  ```bash
  $ python armnn_benchmark.py --tflite ./models/yolov8n_float32.tflite --device GpuAcc --iteration 10
  ```

> `./models/yolov8n_float32.tflite` is for example purposes, you can replace it with your custom tflite model

## Tutorials
By exploring these Python examples, you will be able to fully understand the fundamental steps involved in deploying AI models on these chips, including preprocessing, inference computation, post-processing, and visualization. This will help you quickly grasp the complex workflows and apply them to your own projects.

The available applications and models are listed below, and it is recommended to start with similar functionality to your scenarios.

### List
* **[Deploy Computer Vision Applications Using Ultralytics YOLO](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/ultralytics_tutorial.md)**
* **[Deploy Language Process Applications Using Ollama](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/ollama_tutorial.md)**
* **[Deploy Whisper Applications Using OpenAI Library](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/whisper_tutorial.md)**

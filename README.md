# The Best Implementation for Genio EVKs

<div align="center">
<img src="https://github.com/R300-AI/ITRI-AI-Hub/blob/main/docs/assets/images/pages/genio_510_demonstration_workflow.png" width="780"/>
</div>

## Requirments
* A **Workstation** and **Genio Evaluation Kit**.
* Follow the Developer's [Document](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk.html) to build the **Ubuntu** and **Libraries** on your Genio board.

## How to Use This?
  請依照下列指令將這個範例repository下載到您的Genio板子，然後安裝必要的工作環境(Python=`3.12`)。
  ```bash
  $ git clone https://github.com/R300-AI/MTK-genio-demo.git
  $ cd MTK-genio-demo
  $ pip install -r requirements.txt
  ```

### NeuronRT Benchmarks
  您可以透過以下命令來測試您的模型於MDLA的運算速度
  ```bash
  $ python neuronrt_benchmark.py --tflite ./models/yolov8n_float32.tflite --iteration 10
  ```
### ArmNN Benchmarks
  您可以透過以下命令來測試您的模型於Cortex-A(device=`CpuAcc`)或Mail-G(device=`GpuAcc`)的運算速度
  ```bash
  $ python armnn_benchmark.py --tflite ./models/yolov8n_float32.tflite --device GpuAcc --iteration 10
  ```

> `./models/yolov8n_float32.tflite`為範例之用，你可以將其替換為你自定義的tflite模型

## Demo
By exploring these Python examples, you will be able to fully understand the fundamental steps involved in deploying AI models on these chips, including preprocessing, inference computation, post-processing, and visualization. This will help you quickly grasp the complex workflows and apply them to your own projects.

The available applications and models are listed below, and it is recommended to start with similar functionality of your senarios.
### Tutorial
* **Delegating Computer Vision Applications Using Ultralytics YOLO** | [[ArmNN]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_armnn.md), [[NeuronRT]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_neuronrt.md)

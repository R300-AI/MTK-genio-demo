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
  ```bash
  $ python neuronrt_delegate.py --tflite <path-to-tflite-model> --iteration 10
  ```
### ArmNN Benchmarks
for demo, you can replace `<path-to-tflite-model> ` to `./models/yolov8n_float32.tflite`
  ```bash
  $ python armnn_delegate.py --tflite <path-to-tflite-model> --device GpuAcc --iteration 10
  ```
  > If `AttributeError: 'Delegate' object has no attribute '_library'` occur, please set the `LD_LIBRARY_PATH`:
  > ```bash
  > $ export LD_LIBRARY_PATH=</path/to/ArmNN-linux-aarch64>:$LD_LIBRARY_PATH
  > ```


## Demo
By exploring these Python examples, you will be able to fully understand the fundamental steps involved in deploying AI models on these chips, including preprocessing, inference computation, post-processing, and visualization. This will help you quickly grasp the complex workflows and apply them to your own projects.

The available applications and models are listed below, and it is recommended to start with similar functionality of your senarios.
### Object Detection
* **YOLOv8n Delegation Demo Tutorial** | [[ArmNN]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_armnn.md), [[NeuronRT]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_neuronrt.md)

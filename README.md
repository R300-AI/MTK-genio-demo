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

### ArmNN Benchmarks
  **ArmNN** is a neural network inference library optimized for Arm CPU and GPU. You can use the following command to test the computation speed of your model.
  ```bash
  $ python armnn_benchmark.py --tflite_model ./models/yolov8n_float32.tflite --device GpuAcc --iteration 10
  ```

  > `./models/yolov8n_float32.tflite` is for example purposes, you can replace it with your custom tflite model<br>
  > `--device` options: `CpuAcc` (for Cortex-A),  `GpuAcc` (for Mali-G)

### NeuronRT Benchmarks
  **NeuronRT** is a runtime library for NPUs inference. You can use the following command to preliminarily test the computation speed of your model on MDLA. (**Note**: )
  ```bash
  $ python neuronrt_benchmark.py --tflite_model ./models/yolov8n_float32.tflite --device mdla3.0 --iteration 10
  ```
  > Internet access may be required to compile the tflite model into dla format.<br>
  > `./models/yolov8n_float32.tflite` is for example purposes, you can replace it with your custom tflite model<br>
  > `--device` options: `mdla3.0` (for DLA, G510/700 only),  `mdla2.0` (for DLA, 1200 only), `vpu` (for VPU)*

## Others 
By exploring these Python examples, you will be able to fully understand the fundamental steps involved in deploying AI models on these chips, including preprocessing, inference computation, post-processing, and visualization. This will help you quickly grasp the complex workflows and apply them to your own projects.

The available applications and models are listed below, and it is recommended to start with similar functionality to your scenarios.

### Tutorial List
* **[Deploy Computer Vision Applications Using Ultralytics YOLO](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/ultralytics_tutorial.md)**
* **[Deploy Language Process Applications Using Ollama](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/ollama_tutorial.md)**
* **[Deploy Whisper Applications Using OpenAI Library](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/whisper_tutorial.md)**
* **Optimizing the Performance of AI Streaming Analytics Through Queuing (TO DO)**

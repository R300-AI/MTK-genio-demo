# The Best Implementation for Genio EVKs

<div align="center">
<img src="https://github.com/R300-AI/ITRI-AI-Hub/blob/main/docs/assets/images/pages/genio_510_demonstration_workflow.png" width="780"/>
</div>

## Requirments
* Follow the [Instruction](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk.html) to setup your Genio board with Ubuntu System.

## How to Use This?

By exploring these Python examples, you will be able to fully understand the fundamental steps involved in deploying AI models on these chips, including preprocessing, inference computation, post-processing, and visualization. This will help you quickly grasp the complex workflows and apply them to your own projects.

The available applications and models are listed below, and it is recommended to start with similar functionality of your senarios.

## Benchmarks
### TFLite
```bash
$ pip install -r requirements.txt
$ python test_armnn.py --tflite_model <path_to_tflite_model> --device GpuAcc
```
### DLA
- [How to Converting TFLite Models to DLA Format for NeuronRT](#)
```bash
$ pip install -r requirements.txt
$ python test_neuronrt.py --dla_model <path_to_dla_model> --device GpuAcc
```
## Tutorial
### Object Detection
* **YOLOv8n Delegation Demo Tutorial** | [[ArmNN]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_armnn.md), [[NeuronRT]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_neuronrt.md)

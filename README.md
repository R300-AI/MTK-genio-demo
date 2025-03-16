# The Best Implementation for Genio EVKs

<div align="center">
<img src="https://github.com/R300-AI/ITRI-AI-Hub/blob/main/docs/assets/images/pages/genio_510_demonstration_workflow.png" width="780"/>
</div>

## Requirments
* Follow the [Instruction](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk.html) to install the **Workstation** and flash **Ubuntu** for your Genio board.

# How to Use This?
## Benchmark Tools
for demo, you can replace `<path-to-tflite-model> ` to `./models/yolov8n_float32.tflite`
```bash
$ conda create --name armnn python=3.12 && armnn
$ pip install -r requirements.txt
$ python armnn.py --tflite <path-to-tflite-model> --device GpuAcc --iteration 10
```

## Demo
By exploring these Python examples, you will be able to fully understand the fundamental steps involved in deploying AI models on these chips, including preprocessing, inference computation, post-processing, and visualization. This will help you quickly grasp the complex workflows and apply them to your own projects.

The available applications and models are listed below, and it is recommended to start with similar functionality of your senarios.
### Object Detection
* **YOLOv8n Delegation Demo Tutorial** | [[ArmNN]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_armnn.md), [[NeuronRT]](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/run_yolov8n_via_neuronrt.md)

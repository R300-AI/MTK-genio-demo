# The Best Implementation for Genio EVKs

<div align="center">
<img src="https://github.com/R300-AI/ITRI-AI-Hub/blob/main/docs/assets/images/pages/genio_510_demonstration_workflow.png" width="780"/>
</div>

> [!NOTE]
> * [Config Board](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk.html)
> * [Pretrain or Custom]()


## ArmNN Runtime

```bash
$ conda create --name armnn python=3.9
$ conda activate armnn
```
```bash
$ pip install -r requirments.txt
$ pip install tflite-runtime
```

```bash
$ python run_yolov8n_armnn.py
```

> [!NOTE]
> IF OCCUR DELEGATE ERROR, PLEASE USE THIS
> ```bash
> $ export LD_LIBRARY_PATH=</path/to/ArmNN-linux-aarch64>/build:$LD_LIBRARY_PATH
> ```

## NeuronRT Runtime
```bash
$ sudo neuronrt -a ./models/yolov8n_float32.dla -d
[sudo] password for ubuntu:
```
```bash
$ python run_yolov8n_neuronrt.py
```

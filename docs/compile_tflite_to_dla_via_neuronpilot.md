
NeuronPilot on Workstation is [HERE](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk/step1.html):

```bash
$ conda activate neuronpilot
$ ~/neuronpilot-6.0.5/neuron_sdk/host/bin/ncc-tflite --arch=mdla3.0 --relax-fp32 ./<model_name>_saved_model/<model_name>_float32.tflite ./<model_name>_float32_mdla3.dla
```

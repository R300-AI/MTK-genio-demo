# The Best Implementation for Genio EVKs

<div align="center">
<img src="https://github.com/R300-AI/ITRI-AI-Hub/blob/main/docs/assets/images/pages/genio_510_demonstration_workflow.png" width="780"/>
</div>

## Useful Instructions
* [How to Configure Your Genio EVK: A Beginner's Guide](https://r300-ai.github.io/ITRI-AI-Hub/docs/genio-evk.html)
* [Step-by-Step: Compiling TFLite to DLA Format](#)

## Demo
### ArmNN Runtime

1. Create a new conda environment and activate it.
    ```bash
    $ conda create --name armnn python=3.9
    $ conda activate armnn
    ```

2. Install the required packages.
    ```bash
    $ pip install -r requirments.txt
    $ pip install tflite-runtime
    ```

3. Run the `YOLOv8n_float32.tflite` model with ArmNN.
    ```bash
    $ python run_yolov8n_armnn.py
    ```
    > **Note:** If you encounter a delegate error, set the `LD_LIBRARY_PATH` environment variable as below:
    > ```bash
    > $ export LD_LIBRARY_PATH=</path/to/ArmNN-linux-aarch64>:$LD_LIBRARY_PATH
    > ```

### NeuronRT Runtime 

1. Install the required packages.
    ```bash
    $ pip install -r requirments.txt
    ```

2. Start the NeuronRT runtime with the YOLOv8n model.
    ```bash
    $ sudo neuronrt -a ./models/yolov8n_float32.dla -d
    [sudo] password for ubuntu:
    ```

3. Run the `YOLOv8n_float32.dla` model with NeuronRT.
    ```bash
    $ python run_yolov8n_neuronrt.py
    ```

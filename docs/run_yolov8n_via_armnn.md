### ArmNN Runtime

1. Create a new conda environment for ArmNN and activate it.
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

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

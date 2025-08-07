# Optimizing TFLite Inference Performance with Asynchronous Streaming

## What is Asynchronous?

Asynchronous processing is a method of optimizing task distribution on processors and memory, allowing multiple tasks to execute simultaneously without waiting for each other to complete. In asynchronous streaming inference, the core idea is to divide the process into independent tasks such as data preprocessing, model inference, and result postprocessing. Each task focuses on handling its own input and output, passing data through queues, without needing to worry about the implementation or state of other tasks. This design avoids bottlenecks in the data flow, significantly improving streaming efficiency.

## Implementation of Asynchronous Streaming Inference

### 1. Data Preprocessing (preprocess)

This module is designed as the starting point of the asynchronous process, responsible for reading video frames one by one.

```python
async def preprocess(input_queue, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            await input_queue.put(None)  # Insert None as an end marker when the video ends
            break
        await input_queue.put(frame)  # Add each frame to the input queue
    cap.release()  # Release video resources
```

### 2. Model Inference (predict)

This function module is designed as the intermediate step in the asynchronous process, responsible for retrieving video frames and performing model inference using TFLite.

```python
async def predict(input_queue, output_queue, model):
    while True:
        frame = await input_queue.get()  # Retrieve a frame from the input queue
        if frame is None:
            await output_queue.put(None)  # Insert None as an end marker when receiving the end signal
            break
        results = model.predict(frame, verbose=False)  # Perform inference using the model
        await output_queue.put(results[0].plot())  # Add the inference result to the output queue
```

### 3. Result Postprocessing (postprocess)

This function module is designed as the endpoint of the asynchronous process, responsible for presenting the inference results to the user and providing real-time processing speed information.

```python
async def postprocess(output_queue):
    while True:
        start_time = time.time()  # Record the start time
        result = await output_queue.get()  # Retrieve the inference result from the output queue
        if result is None:
            break
        cv2.imshow('streaming', result)  # Display the inference result
        print('Streaming Speed:', (time.time() - start_time) * 1000, 'ms')  # Calculate processing speed
        if cv2.waitKey(1) == ord('q'):  # Exit when the 'q' key is pressed
            break
    cv2.destroyAllWindows()  # Close all OpenCV windows
```

### Main Program (main)

Define the above functions as an asynchronous process with three modules: data preprocessing, model inference, and result display. These modules communicate through queues, are decoupled from each other, and execute independently, enabling an efficient asynchronous streaming inference process.

```python
async def main():
    input_queue = asyncio.Queue()  # Initialize the input queue
    output_queue = asyncio.Queue()  # Initialize the output queue
    cap = cv2.VideoCapture("./data/serve.mp4")  # Open the video file
    model = YOLO("./models/yolov8n_float32.tflite", task='detect')  # Load the YOLO model
    
    await asyncio.gather(
        preprocess(input_queue, cap),  # Start the data preprocessing task
        predict(input_queue, output_queue, model),  # Start the model inference task
        postprocess(output_queue)  # Start the result postprocessing task
    )
```

## Run the Asynchronous Inference

By splitting data preprocessing, model inference, and result display into independent tasks and passing data through queues, asynchronous streaming inference achieves efficient parallel processing. This design fully utilizes hardware resources. You can run the asynchronous streaming inference with the following command:

```bash
python ultralytics_async.py
```
from ultralytics import YOLO
import cv2, asyncio, time

async def preprocess(input_queue, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            await input_queue.put(None)
            break
        await input_queue.put(frame)
    cap.release() 

async def predict(input_queue, output_queue, model):
    while True:
        frame = await input_queue.get()
        if frame is None:
            await output_queue.put(None)
            break
        results = model.predict(frame, verbose=False)
        await output_queue.put(results[0].plot())
    
async def postprocess(output_queue):
    while True:
        start_time = time.time()
        result = await output_queue.get()
        if result is None:
            break
        cv2.imshow('streaming', result)
        print('Streaming Speed:', (time.time() - start_time) * 1000, 'ms')
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
         
async def main():
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    cap = cv2.VideoCapture("./data/serve.mp4")
    model = YOLO("./models/yolov8n_float32.tflite", task='detect')
    
    await asyncio.gather(
        preprocess(input_queue, cap),
        predict(input_queue, output_queue, model),
        postprocess(output_queue)
    )
    
asyncio.run(main())





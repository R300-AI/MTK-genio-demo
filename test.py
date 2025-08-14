[main] 開始處理影片: ./data/video.mp4
[inference] start
[inference] YOLO model loaded
[inference] got item from input_queue: 0
[inference] created predict task for frame 0
[predict] start: frame 0
[inference] got item from input_queue: 1
[inference] created predict task for frame 1
[predict] start: frame 1
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 2
[inference] created predict task for frame 2
[predict] start: frame 2
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 3
[inference] created predict task for frame 3
[predict] start: frame 3
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 4
[inference] created predict task for frame 4
[predict] start: frame 4
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 5
[inference] created predict task for frame 5
[predict] start: frame 5
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 6
[inference] created predict task for frame 6
[predict] start: frame 6
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 7
[inference] created predict task for frame 7
[predict] start: frame 7
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 8
[inference] created predict task for frame 8
[predict] start: frame 8
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
[inference] got item from input_queue: 9
[inference] created predict task for frame 9
[predict] start: frame 9
[inference] got item from input_queue: 10
[inference] created predict task for frame 10
[predict] start: frame 10
Ultralytics 8.3.175 🚀 Python-3.12.2 torch-2.6.0+cpu CPU (Cortex-A55)
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
/home/ubuntu/miniconda3/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
Successfully loaded NeuronRT delegate for DLA inference
/home/ubuntu/miniconda3/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
Successfully loaded NeuronRT delegate for DLA inference
Load DLA model: ./models/yolov8n_float32.dla
[predict] done: frame 8
[predict] put to output_queue: frame 8
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.
[inference] got item from input_queue: 11
                                         [inference] created predict task for frame 11
                                                                                      [predict] start: frame 11
                                                                                                               [predict] done: frame 1
[predict] put to output_queue: frame 1
[inference] got item from input_queue: 12
[inference] created predict task for frame 12
[predict] start: frame 12
[predict] done: frame 7
[predict] put to output_queue: frame 7
[inference] got item from input_queue: 13
[inference] created predict task for frame 13
[predict] start: frame 13
[predict] done: frame 6
[predict] put to output_queue: frame 6
[inference] got item from input_queue: 14
[inference] created predict task for frame 14
[predict] start: frame 14
[predict] done: frame 9
[predict] put to output_queue: frame 9

... ...
[inference] got item from input_queue: 107
[inference] created predict task for frame 107
[predict] start: frame 107
[predict] done: frame 97
[predict] put to output_queue: frame 97
[inference] got item from input_queue: 108
[inference] created predict task for frame 108
[predict] start: frame 108
[predict] done: frame 98
[predict] put to output_queue: frame 98
[inference] got item from input_queue: None
[inference] input_queue end signal received
[inference] end



from ultralytics import YOLO
import cv2, asyncio, time, argparse

async def preprocess(input_queue: asyncio.Queue, video_capture: cv2.VideoCapture):
    #print('[preprocess] start')
    loop = asyncio.get_running_loop()
    index = 0
    while True:
        ret, frame = await loop.run_in_executor(None, video_capture.read)
        #print(f'[preprocess] read frame {index}, ret={ret}')
        await input_queue.put((index, frame, time.time()) if ret else None)
        if not ret:
            #print('[preprocess] end of video')
            break
        index += 1
    video_capture.release()
    #print('[preprocess] end')

async def predict(index, frame, capture_time):
    print(f'[predict] start: frame {index}')
    result = await asyncio.to_thread(lambda: model.predict(frame, verbose=False)[0])
    print(f'[predict] done: frame {index}')
    await output_queue.put((index, result, capture_time, time.time()))
    print(f'[predict] put to output_queue: frame {index}')

async def inference(input_queue: asyncio.Queue, output_queue: asyncio.Queue, model_path: str):
    print('[inference] start')
    model = YOLO(model_path, task='detect')
    print('[inference] YOLO model loaded')
    while True:
        item = await input_queue.get()
        print(f'[inference] got item from input_queue: {item[0] if item else None}')
        if item is None:
            print('[inference] input_queue end signal received')
            await output_queue.put(None)
            break
        index, frame, capture_time = item
        asyncio.create_task(predict(index, frame, capture_time))
        print(f'[inference] created predict task for frame {index}')
    print('[inference] end')

async def postprocess(output_queue: asyncio.Queue, show_fps=True):
    #print('[postprocess] start')
    while True:
        item = await output_queue.get()
        #print(f'[postprocess] got item from output_queue: {item[0] if item else None}')
        if item is None:
            #print('[postprocess] output_queue end signal received')
            break
        index, result, capture_time, predict_time = item
        result = result.plot()
        if show_fps:
            dt = max(1e-6, time.time() - capture_time)
            fps = 1.0 / dt
            cv2.putText(result, f'FPS:{fps:.1f}  Frame:{index}',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('YOLO Detection Stream', cv2.resize(result, (720, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('[postprocess] user quit')
            break
    cv2.destroyAllWindows()
    #print('[postprocess] end')

async def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO 非同步串流推論')
    parser.add_argument('--video_path', type=str, default='./data/video.mp4')
    parser.add_argument('--tflite_model', type=str, default='./models/yolov8n_float32.tflite')
    args = parser.parse_args()

    video_capture = cv2.VideoCapture(args.video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video_path}')

    input_queue  = asyncio.Queue()
    output_queue = asyncio.Queue()

    print(f"[main] 開始處理影片: {args.video_path}")

    await asyncio.gather(
        preprocess(input_queue, video_capture),
        inference(input_queue, output_queue, args.tflite_model),
        postprocess(output_queue)
    )

if __name__ == '__main__':
    asyncio.run(main())

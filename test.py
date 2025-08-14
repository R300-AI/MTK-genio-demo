
from ultralytics import YOLO
import cv2, asyncio, time, argparse

async def preprocess(input_queue: asyncio.Queue, video_capture: cv2.VideoCapture):
    print('preprocess start')
    loop = asyncio.get_running_loop()
    index = 0
    while True:
        ret, frame = await loop.run_in_executor(None, video_capture.read)
        await input_queue.put((index, frame, time.time()) if ret else None)
        if not ret:
            break
        index += 1
    video_capture.release()
    print('preprocess end')

async def predict(index, frame, capture_time):
    result = await asyncio.to_thread(lambda: model.predict(frame)[0])
    await output_queue.put((index, result, capture_time, time.time()))

async def inference(input_queue: asyncio.Queue, output_queue: asyncio.Queue, model_path: str):
    print('inference start')
    model = YOLO(model_path, task='detect')
    while True:
        item = await input_queue.get()
        if item is None:
            await output_queue.put(None)
            break
        index, frame, capture_time = item
        asyncio.create_task(predict(index, frame, capture_time))
    print('inference end')

async def postprocess(output_queue: asyncio.Queue, show_fps=True):
    print('postprocess start')
    while True:
        item = await output_queue.get()
        if item is None:
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
            break
    cv2.destroyAllWindows()
    print('postprocess end')

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

    print(f"開始處理影片: {args.video_path}")

    await asyncio.gather(
        preprocess(input_queue, video_capture),
        inference(input_queue, output_queue, args.tflite_model),
        postprocess(output_queue)
    )

if __name__ == '__main__':
    asyncio.run(main())

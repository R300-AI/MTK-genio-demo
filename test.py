from ultralytics import YOLO
import cv2, asyncio, time, argparse, psutil

# ----------------------------
# Producer: read frames
# ----------------------------
async def preprocess(input_queue: asyncio.Queue, video_capture: cv2.VideoCapture, num_sentinels: int):
    print('preprocess start')
    loop = asyncio.get_running_loop()
    index = 0
    while True:
        ret, frame = await loop.run_in_executor(None, video_capture.read)  # non-blocking for event loop
        if not ret:
            break
        # put (index, frame, capture_time)
        await input_queue.put((index, frame, time.time()))
        index += 1
    video_capture.release()
    # send one sentinel per worker so each exits
    for _ in range(num_sentinels):
        await input_queue.put(None)
    print('preprocess end')

# ----------------------------
# Concurrent predictor workers
# ----------------------------
async def predict_worker(worker_id: int,
                           input_queue: asyncio.Queue,
                           output_queue: asyncio.Queue,
                           model_path: str):
    mem = psutil.virtual_memory()
    if mem.percent >= 90:
        print(f"predictor[{worker_id}] 啟動時記憶體已超過 90%，自動跳過此 worker。")
        return
    print(f'predictor[{worker_id}] start')
    model = YOLO(model_path, task='detect')  # own instance per worker
    while True:
        item = await input_queue.get()
        if item is None:  # sentinel
            await output_queue.put(None)  # forward sentinel
            break
        index, frame, capture_time = item

        # Offload blocking predict to a thread
        result = await asyncio.to_thread(
            lambda: model.predict(frame)[0]
        )
        await output_queue.put((index, result, capture_time, worker_id, time.time()))
    print(f'predictor[{worker_id}] end')

# ----------------------------
# Consumer: display results
# ----------------------------
async def postprocess(output_queue: asyncio.Queue, num_sentinels: int, show_fps=True):
    print('postprocess start')
    num_finished = 0
    while True:
        item = await output_queue.get()
        if item is None:
            num_finished += 1
            if num_finished >= num_sentinels:
                break
            continue
        index, result, capture_time, worker_id, predict_time = item
        result = result.plot()
        if show_fps:
            dt = max(1e-6, time.time() - capture_time)
            fps = 1.0 / dt
            cv2.putText(result, f'Worker{worker_id}  FPS:{fps:.1f}  Frame:{index}',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('YOLO Detection Stream (concurrent)', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print('postprocess end')

# ----------------------------
# Orchestrator
# ----------------------------
async def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO 非同步並行推論')
    parser.add_argument('--video_path', type=str, default='./data/serve.mp4')
    parser.add_argument('--tflite_model', type=str, default='./models/yolov8n_float32.tflite')
    parser.add_argument('--async_workers', type=int, default=1, help='並行推論 worker 數')
    args = parser.parse_args()

    video_capture = cv2.VideoCapture(args.video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video_path}')

    input_queue  = asyncio.Queue()
    output_queue = asyncio.Queue()

    print(f"開始處理影片: {args.video_path}  | workers={args.async_workers}")

    # Build tasks
    producers = [asyncio.create_task(preprocess(input_queue, video_capture, args.async_workers))]
    predictors = [
        asyncio.create_task(predict_worker(i, input_queue, output_queue, args.tflite_model))
        for i in range(args.async_workers)
    ]
    consumers = [asyncio.create_task(postprocess(output_queue, args.async_workers))]

    # Run all
    await asyncio.gather(*producers, *predictors, *consumers)

if __name__ == '__main__':
    asyncio.run(main())

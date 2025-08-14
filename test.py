from ultralytics import YOLO
import cv2, asyncio, time, argparse
from typing import Tuple, Optional

# ----------------------------
# Producer: read frames
# ----------------------------
async def preprocess(q_in: asyncio.Queue, cap: cv2.VideoCapture, n_sentinels: int):
    print('preprocess start')
    loop = asyncio.get_running_loop()
    seq = 0
    while True:
        ret, frame = await loop.run_in_executor(None, cap.read)  # non-blocking for event loop
        if not ret:
            break
        # put (sequence_id, frame, t_capture)
        await q_in.put((seq, frame, time.time()))
        seq += 1
    cap.release()
    # send one sentinel per worker so each exits
    for _ in range(n_sentinels):
        await q_in.put(None)
    print('preprocess end')

# ----------------------------
# Concurrent predictor workers
# ----------------------------
async def predictor_worker(worker_id: int,
                           q_in: asyncio.Queue,
                           q_out: asyncio.Queue,
                           model_path: str):
    print(f'predictor[{worker_id}] start')
    model = YOLO(model_path, task='detect')  # own instance per worker
    while True:
        item: Optional[Tuple[int, 'np.ndarray', float]] = await q_in.get()
        if item is None:  # sentinel
            await q_out.put(None)  # forward sentinel
            break
        seq, frame, t_cap = item

        # Offload blocking predict to a thread
        result = await asyncio.to_thread(
            lambda: model.predict(frame, stream=False, verbose=False)[0]
        )
        vis = result.plot()
        await q_out.put((seq, vis, t_cap, worker_id, time.time()))
    print(f'predictor[{worker_id}] end')

# ----------------------------
# Consumer: display results
# ----------------------------
async def postprocess(q_out: asyncio.Queue, n_sentinels: int, show_fps=True):
    print('postprocess start')
    done = 0
    while True:
        item = await q_out.get()
        if item is None:
            done += 1
            if done >= n_sentinels:
                break
            continue

        seq, img, t_cap, worker_id, t_pred = item
        if show_fps:
            dt = max(1e-6, time.time() - t_cap)
            fps = 1.0 / dt
            cv2.putText(img, f'W{worker_id}  FPS:{fps:.1f}  seq:{seq}',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('YOLO Detection Stream (concurrent)', img)
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
    parser.add_argument('--model', type=str, default='./models/yolov8n_float32.tflite')
    parser.add_argument('--predict_workers', type=int, default=2, help='並行推論 worker 數')
    parser.add_argument('--queue_size', type=int, default=4, help='輸入佇列長度（背壓）')
    parser.add_argument('--latest_only', action='store_true',
                        help='若輸入佇列滿則丟棄最舊幀（降低延遲）')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video_path}')

    q_in  = asyncio.Queue(maxsize=args.queue_size)
    q_out = asyncio.Queue(maxsize=args.queue_size * args.predict_workers)

    print(f"開始處理影片: {args.video_path}  | workers={args.predict_workers}")

    # Optional: a small helper to push frames with "latest only" policy
    async def preprocess_with_policy():
        loop = asyncio.get_running_loop()
        seq = 0
        while True:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                break
            item = (seq, frame, time.time())
            if args.latest_only and q_in.full():
                try:
                    _ = q_in.get_nowait()   # drop oldest
                    q_in.task_done()
                except asyncio.QueueEmpty:
                    pass
            await q_in.put(item)
            seq += 1
        cap.release()
        for _ in range(args.predict_workers):
            await q_in.put(None)

    # Build tasks
    producers = [asyncio.create_task(preprocess_with_policy())]
    predictors = [
        asyncio.create_task(predictor_worker(i, q_in, q_out, args.model))
        for i in range(args.predict_workers)
    ]
    consumers = [asyncio.create_task(postprocess(q_out, args.predict_workers))]

    # Run all
    await asyncio.gather(*producers, *predictors, *consumers)

if __name__ == '__main__':
    asyncio.run(main())

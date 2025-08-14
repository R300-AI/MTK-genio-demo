from ultralytics import YOLO
import cv2, asyncio, time, argparse

# ---------- 協程區 ----------
async def preprocess(cap, input_queue):
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            await input_queue.put(None)
            break
        await input_queue.put((index, frame, time.time()))
        index += 1

async def predict(input_queue, output_queue, model):
    async def run_predict(index, frame, t0):
        # 推論放到 thread pool，避免阻塞事件迴圈
        result = await asyncio.to_thread(lambda: model.predict(frame, verbose=False)[0])
        await output_queue.put((index, result, t0, time.time()))

    while True:
        item = await input_queue.get()
        if item is None:
            await output_queue.put(None)
            break
        index, frame, t0 = item
        asyncio.create_task(run_predict(index, frame, t0))  # 無上限並行

async def postprocess(output_queue):
    results_buffer = {}
    next_index = 0

    while True:
        item = await output_queue.get()
        if item is None:
            break
        index, result, t0, t1 = item
        results_buffer[index] = (result, t0, t1)

        # 確保順序輸出
        while next_index in results_buffer:
            result, t0, t1 = results_buffer.pop(next_index)
            annotated = result.plot()
            dt = (t1 - t0) * 1000
            cv2.imshow("YOLO Stream", annotated)
            print(f"Frame {next_index}: {dt:.1f} ms")
            next_index += 1

            if cv2.waitKey(1) == 27:  # ESC
                return

# ---------- 主程式 ----------
def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO 非同步串流推論')
    parser.add_argument('--video_path', type=str, default='./data/video.mp4')
    parser.add_argument('--tflite_model', type=str, default='./models/yolov8n_float32.tflite')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video_path}')

    input_queue  = asyncio.Queue()
    output_queue = asyncio.Queue()

    print(f"[main] 開始處理影片: {args.video_path}")
    model = YOLO(args.tflite_model, task='detect')

    loop = asyncio.get_event_loop()
    tasks = [
        preprocess(cap, input_queue),
        predict(input_queue, output_queue, model),
        postprocess(output_queue)
    ]
    loop.run_until_complete(asyncio.gather(*tasks))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

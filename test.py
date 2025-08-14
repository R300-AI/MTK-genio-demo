from ultralytics import YOLO
import cv2, asyncio, argparse, concurrent.futures, pickle

# ------------------ 子進程推論函式 ------------------
def run_inference(frame_bytes, model_path):
    frame = pickle.loads(frame_bytes)
    model = YOLO(model_path, task="detect")
    result = model.predict(frame, stream=False)[0]
    return pickle.dumps(result)

# ------------------ 前處理 ------------------
async def preprocess(input_queue, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        await input_queue.put(frame)
    await input_queue.put(None)
    cap.release()

# ------------------ 推論（在 coroutine 裡呼叫子進程） ------------------
async def inference(input_queue, output_queue, model_path):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        while True:
            frame = await input_queue.get()
            if frame is None:
                break
            frame_bytes = pickle.dumps(frame)
            result_bytes = await loop.run_in_executor(executor, run_inference, frame_bytes, model_path)
            result = pickle.loads(result_bytes)
            await output_queue.put(result)
    await output_queue.put(None)

# ------------------ 後處理 ------------------
async def postprocess(output_queue):
    while True:
        result = await output_queue.get()
        if result is None:
            break
        img = result.plot()
        cv2.imshow("Ultralytics Async Inference", cv2.resize(img, (720, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# ------------------ 主流程 ------------------
async def main(args):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    input_queue = asyncio.Queue(maxsize=30)
    output_queue = asyncio.Queue(maxsize=30)

    await asyncio.gather(
        preprocess(input_queue, cap),
        inference(input_queue, output_queue, args.tflite_model),
        postprocess(output_queue)
    )

# ------------------ CLI 入口 ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Async Inference with Subprocess")
    parser.add_argument("--video_path", type=str, default="./data/video.mp4")
    parser.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
    args = parser.parse_args()

    asyncio.run(main(args))
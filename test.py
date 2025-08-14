from ultralytics import YOLO
import cv2, asyncio, time, argparse

# ---------- 協程區 ----------
async def preprocess(cap, input_queue):
    index = 0
    print("[preprocess] 啟動")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[preprocess] 影片讀取結束")
            await input_queue.put(None)
            break
        t0 = time.time()
        print(f"[preprocess] 推入 frame {index} at {t0:.3f}")
        await input_queue.put((index, frame, t0))
        index += 1

async def predict(input_queue, output_queue, model):
    sem = asyncio.Semaphore(1)  # 對 DLA/TFLite 建議 1
    print("[predict] 啟動")

    async def run_predict(index, frame, t0):
        async with sem:
            print(f"[predict] 開始推論 frame {index} at {time.time():.3f}")
            # 放到 thread pool 避免阻塞
            result = await asyncio.to_thread(lambda: model.predict(frame, verbose=False)[0])
            t1 = time.time()
            print(f"[predict] 完成推論 frame {index} at {t1:.3f}, 耗時 {(t1-t0)*1000:.1f} ms")
        await output_queue.put((index, result, t0, t1))

    while True:
        item = await input_queue.get()
        if item is None:
            print("[predict] 收到結束訊號")
            await output_queue.put(None)
            break
        index, frame, t0 = item
        asyncio.create_task(run_predict(index, frame, t0))

async def postprocess(output_queue):
    results_buffer = {}
    next_index = 0
    print("[postprocess] 啟動")

    while True:
        item = await output_queue.get()
        if item is None:
            print("[postprocess] 收到結束訊號")
            break
        index, result, t0, t1 = item
        results_buffer[index] = (result, t0, t1)
        print(f"[postprocess] 收到 frame {index} 結果 at {time.time():.3f}")

        # 確保順序輸出
        while next_index in results_buffer:
            result, t0, t1 = results_buffer.pop(next_index)
            annotated = result.plot()
            dt = (t1 - t0) * 1000
            print(f"[postprocess] 顯示 frame {next_index}, total latency {dt:.1f} ms")
            cv2.imshow("YOLO Stream", annotated)
            next_index += 1
            if cv2.waitKey(1) == 27:  # ESC
                print("[postprocess] ESC pressed, 結束")
                return

# ---------- 主程式 ----------
def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO 非同步串流推論（改良版）')
    parser.add_argument('--video_path', type=str, default='./data/video.mp4')
    parser.add_argument('--tflite_model', type=str, default='./models/yolov8n_float32.tflite')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video_path}')

    input_queue  = asyncio.Queue()
    output_queue = asyncio.Queue()

    print(f"[main] 開始處理影片: {args.video_path}")
    print("[main] 初始化模型一次（YOLO / TFLite / DLA）...")
    model = YOLO(args.tflite_model, task='detect')
    print("[main] 模型初始化完成")

    async def runner():
        await asyncio.gather(
            preprocess(cap, input_queue),
            predict(input_queue, output_queue, model),
            postprocess(output_queue)
        )

    asyncio.run(runner())

    cap.release()
    cv2.destroyAllWindows()
    print("[main] 全部任務完成")

if __name__ == "__main__":
    main()

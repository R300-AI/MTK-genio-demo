import os
import cv2
import time
import asyncio
import argparse
from ultralytics import YOLO

# 固定參數（不再讓使用者輸入）
CONCURRENCY = 1
QSIZE = 4
LOG_EVERY = 30

# ------------------ 協程 ------------------
async def preprocess(cap, input_q: asyncio.Queue):
    print("[preprocess] 啟動")
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[preprocess] 影片讀取結束，送出結束訊號")
            await input_q.put(None)
            break
        t0 = time.time()
        await input_q.put((index, frame, t0))
        if index % LOG_EVERY == 0:
            print(f"[preprocess] 推入 frame {index} at {t0:.3f} (queue size≈{input_q.qsize()})")
        await asyncio.sleep(0)  # 每幀都讓出控制權
        index += 1


async def predict(input_q: asyncio.Queue, output_q: asyncio.Queue, model: YOLO):
    print(f"[predict] 啟動，concurrency={CONCURRENCY}")
    sem = asyncio.Semaphore(CONCURRENCY) if CONCURRENCY > 0 else None
    inflight = set()

    async def run_predict(idx, frame, t0):
        if sem:
            async with sem:
                t_start = time.time()
                print(f"[predict] ▶ 開始推論 frame {idx} at {t_start:.3f}")
                result = await asyncio.to_thread(lambda: model.predict(frame, verbose=False)[0])
                t1 = time.time()
                print(f"[predict] ✓ 完成推論 frame {idx} at {t1:.3f}, 耗時 {(t1-t_start)*1000:.1f} ms")
        else:
            t_start = time.time()
            result = await asyncio.to_thread(lambda: model.predict(frame, verbose=False)[0])
            t1 = time.time()
        await output_q.put((idx, result, t0, t1))

    while True:
        item = await input_q.get()
        if item is None:
            print(f"[predict] 收到結束訊號，等待 {len(inflight)} 個在途推論完成...")
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
            await output_q.put(None)
            break
        idx, frame, t0 = item
        task = asyncio.create_task(run_predict(idx, frame, t0))
        inflight.add(task)
        task.add_done_callback(lambda t: inflight.discard(t))


async def postprocess(output_q: asyncio.Queue, display: bool):
    print(f"[postprocess] 啟動，display={display}")
    buffer = {}
    next_idx = 0

    def render(res):
        return res.plot()

    while True:
        item = await output_q.get()
        if item is None:
            while next_idx in buffer:
                res, t0, t1 = buffer.pop(next_idx)
                frame = render(res)
                if display:
                    cv2.imshow("YOLO Stream", cv2.resize(frame, (720, 480)))
                    if cv2.waitKey(1) == 27:
                        return
                print(f"[postprocess] 顯示 frame {next_idx}, total latency {(t1-t0)*1000:.1f} ms")
                next_idx += 1
            break

        idx, res, t0, t1 = item
        buffer[idx] = (res, t0, t1)
        while next_idx in buffer:
            res, t0, t1 = buffer.pop(next_idx)
            frame = render(res)
            if display:
                cv2.imshow("YOLO Stream", frame)
                if cv2.waitKey(1) == 27:
                    return
            print(f"[postprocess] 顯示 frame {next_idx}, total latency {(t1-t0)*1000:.1f} ms")
            next_idx += 1

# ------------------ 主流程平鋪 ------------------
parser = argparse.ArgumentParser(description='Ultralytics YOLO 非同步串流推論（Async Streaming Demo）')
parser.add_argument('--video_path', type=str, default='./data/video.mp4')
parser.add_argument('--tflite_model', type=str, default='./models/yolov8n_float32.tflite')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    raise RuntimeError(f'Cannot open video: {args.video_path}')

input_queue = asyncio.Queue(maxsize=QSIZE)
output_queue = asyncio.Queue(maxsize=QSIZE)

can_display = bool(os.environ.get("DISPLAY"))

print(f"[main] 開始處理影片: {args.video_path}")
print("[main] 初始化模型一次（YOLO / TFLite / DLA）...")
model = YOLO(args.tflite_model, task='detect')
print("[main] 模型初始化完成")

asyncio.run(asyncio.gather(
    preprocess(cap, input_queue),
    predict(input_queue, output_queue, model),
    postprocess(output_queue, display=can_display)
))

cap.release()
if can_display:
    cv2.destroyAllWindows()
print("[main] 全部任務完成")
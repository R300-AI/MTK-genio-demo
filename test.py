import os
import cv2
import time
import asyncio
import argparse
from ultralytics import YOLO

# ------------------ 協程：preprocess / predict / postprocess ------------------
async def preprocess(cap, input_q: asyncio.Queue, log_every: int = 30):
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
        if index % log_every == 0:
            print(f"[preprocess] 推入 frame {index} at {t0:.3f} (queue size≈{input_q.qsize()})")

        # 讓出控制，避免單一協程長時間霸佔
        if index % 10 == 0:
            await asyncio.sleep(0)

        index += 1


async def predict(input_q: asyncio.Queue, output_q: asyncio.Queue, model: YOLO, concurrency: int = 1):
    print(f"[predict] 啟動，concurrency={concurrency}")
    sem = asyncio.Semaphore(concurrency) if concurrency > 0 else None
    inflight = set()
    started = 0

    async def run_predict(idx, frame, t0):
        nonlocal started
        if sem:
            async with sem:
                t_start = time.time()
                print(f"[predict] ▶ 開始推論 frame {idx} at {t_start:.3f} (inflight before={len(inflight)})")
                result = await asyncio.to_thread(lambda: model.predict(frame, verbose=False)[0])
                t1 = time.time()
                print(f"[predict] ✓ 完成推論 frame {idx} at {t1:.3f}, 推論耗時 {(t1 - t_start) * 1000:.1f} ms")
        else:
            t_start = time.time()
            print(f"[predict] ▶ 開始推論 frame {idx} at {t_start:.3f} (inflight before={len(inflight)})")
            result = await asyncio.to_thread(lambda: model.predict(frame, verbose=False)[0])
            t1 = time.time()
            print(f"[predict] ✓ 完成推論 frame {idx} at {t1:.3f}, 推論耗時 {(t1 - t_start) * 1000:.1f} ms")

        await output_q.put((idx, result, t0, t1))

    while True:
        item = await input_q.get()
        if item is None:
            print(f"[predict] 收到結束訊號，等待 {len(inflight)} 個在途推論完成...")
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
            print("[predict] 在途推論已清空 → 發送結束訊號給 postprocess")
            await output_q.put(None)
            break

        idx, frame, t0 = item
        task = asyncio.create_task(run_predict(idx, frame, t0))
        inflight.add(task)
        task.add_done_callback(inflight.discard)
        started += 1

        # 週期性 log
        if started % 20 == 0:
            print(f"[predict] 已建立 {started} 個推論任務，inflight={len(inflight)}")


async def postprocess(output_q: asyncio.Queue, display: bool, writer: cv2.VideoWriter | None):
    print(f"[postprocess] 啟動，display={display}, writer={'on' if writer else 'off'}")
    buffer = {}
    next_idx = 0

    def render(res):
        # Ultralytics: result.plot() 會回傳已標註影像
        return res.plot()

    while True:
        item = await output_q.get()
        if item is None:
            print(f"[postprocess] 收到結束訊號，flush buffer（當前 {len(buffer)} 筆）")
            # Flush remaining
            while next_idx in buffer:
                res, t0, t1 = buffer.pop(next_idx)
                frame = render(res)
                if display:
                    cv2.imshow("YOLO Stream", frame)
                    if cv2.waitKey(1) == 27:
                        print("[postprocess] ESC pressed，提前結束")
                        return
                if writer:
                    writer.write(frame)
                print(f"[postprocess] 顯示 frame {next_idx}, total latency {(t1 - t0) * 1000:.1f} ms")
                next_idx += 1
            print("[postprocess] 完成 flush，結束")
            break

        idx, res, t0, t1 = item
        buffer[idx] = (res, t0, t1)
        # 依序輸出
        while next_idx in buffer:
            res, t0, t1 = buffer.pop(next_idx)
            frame = render(res)
            if display:
                cv2.imshow("YOLO Stream", frame)
                if cv2.waitKey(1) == 27:
                    print("[postprocess] ESC pressed，提前結束")
                    return
            if writer:
                writer.write(frame)
            print(f"[postprocess] 顯示 frame {next_idx}, total latency {(t1 - t0) * 1000:.1f} ms")
            next_idx += 1


# ------------------ 主流程 ------------------
async def amain(args):
    # 開啟影片
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    # 建立背壓 queue，避免 preprocess 單向灌爆
    input_q = asyncio.Queue(maxsize=args.qsize)
    output_q = asyncio.Queue(maxsize=args.qsize)

    # 視窗顯示/輸出
    can_display = (not args.headless) and bool(os.environ.get("DISPLAY"))
    writer = None
    if args.save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_path, fourcc, fps, (w, h))
        print(f"[main] 將輸出儲存到 {args.save_path}，fps={fps}, size=({w},{h})")

    # 模型只初始化一次
    print("[main] 初始化模型一次（YOLO / TFLite / DLA）...")
    model = YOLO(args.tflite_model, task="detect")
    print("[main] 模型初始化完成")

    tasks = [
        preprocess(cap, input_q, log_every=args.log_every),
        predict(input_q, output_q, model, concurrency=args.concurrency),
        postprocess(output_q, display=can_display, writer=writer),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        cap.release()
        if writer:
            writer.release()
        if can_display:
            cv2.destroyAllWindows()
        print("[main] 全部任務完成")


def parse_args():
    p = argparse.ArgumentParser(description="Ultralytics YOLO 非同步串流推論（修正版）")
    p.add_argument("--video_path", type=str, default="./data/video.mp4")
    p.add_argument("--tflite_model", type=str, default="./models/yolov8n_float32.tflite")
    p.add_argument("--concurrency", type=int, default=1, help="TFLite/DLA 建議先用 1")
    p.add_argument("--qsize", type=int, default=8, help="pipeline 背壓 queue 大小")
    p.add_argument("--headless", action="store_true", help="強制關閉視窗（例如 SSH/沒有 DISPLAY）")
    p.add_argument("--save_path", type=str, default="", help="輸出結果影片檔（例如 out.mp4）")
    p.add_argument("--log_every", type=int, default=30, help="preprocess 推入 log 的間隔")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[main] 開始處理影片: {args.video_path}")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
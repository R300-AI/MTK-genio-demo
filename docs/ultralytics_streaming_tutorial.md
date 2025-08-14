
# Ultralytics YOLO 即時串流推論教學

本文件將詳細說明如何在 MediaTek Genio 平台上配置並運行 Ultralytics YOLO 的高效能即時串流推論系統。


## 系統配置與前置作業

### 第一步：理解 Ultralytics 推論後端機制

請開啟 `ultralytics/nn/autobackend.py`，在 TFLite 區段（約第 465-490 行）您會看到以下程式碼：

```python
else:  # TFLite
    LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
    
    # === 原始 TFLite 解釋器 ===
    interpreter = Interpreter(model_path=w)
    
    # === 錯誤提示：需要手動配置後端 ===
    raise RuntimeError(
        f"Genio Backend not configured! Please edit {__file__} and uncomment one of the backend options above. Please see the tutorial at docs/ultralytics_streaming_tutorial.md for detailed instructions."
    )
    
    # === 選項 A: 使用 ArmNN 加速 (CPU/ GPU) ===
    # import tensorflow as tf
    #
    # interpreter = tf.lite.Interpreter(
    #     model_path=w,
    #     experimental_delegates=[
    #         armnn_delegate = tf.lite.experimental.load_delegate(
    #             library="<path to libarmnnDelegate.so>",
    #             options={"backends": "<CpuAcc or GpuAcc>", "logging-severity": "fatal"}
    #         )
    #     ]
    # )
    # LOGGER.info("Successfully loaded ArmNN delegate for TFLite inference")

    # === 選項 B: 使用 NeuronRT 加速(MDLA/ VPU) ===
    # from utils.neuronpilot.neuronrt import Interpreter
    # 
    # interpreter = Interpreter(
    #     tflite_path=w, 
    #     dla_path="<path to your dla model>",       
    #     device= "<mdla3.0, mdla2.0 or vpu>"
    # )
    # LOGGER.info("Successfully loaded NeuronRT delegate for DLA inference")
```



### 第二步：手動配置推論後端

根據您的硬體與需求選擇加速方案，並將對應區塊的程式碼取消註解並填入正確參數：

### 選擇ArmNN Delegate（CPU/GPU 加速）

1. 取消註解 **選項 A** 相關的程式碼。
2. 將 `<path to libarmnnDelegate.so>` 參數改為您系統上 ArmNN `libarmnnDelegate.so` 的實際路徑。
3. 將 `<CpuAcc or GpuAcc>` 參數設為 `"CpuAcc"` 或 `"GpuAcc"`。

### 選擇NeuronRT Delegate（MDLA/VPU 加速）

1. 取消註解 **選項 B** 相關的程式碼。
2. 將 `<path to your dla model>` 改為您的 DLA model 路徑。
3. `<mdla3.0, mdla2.0 or vpu>` 參數請依照您的硬體設為 `"mdla3.0"`、`"mdla2.0"` 或 `"vpu"`。

> **請務必將「原生 TFLite 解譯器」的 `interpreter = Interpreter(model_path=w)` 以及 `raise RuntimeError(...)` 這兩行註解掉，避免重複執行或出現錯誤。**


### 第三步：驗證推論後端配置

完成上述步驟後，Ultralytics 就會使用您選擇的推論後端進行加速推論。

```python
from ultralytics import YOLO

model = YOLO("./models/yolov8n_float32.tflite")
results = model.predict(["./data/bus.jpg"])
results[0].show()
```

---

## 非同步化串流推論


即時物件偵測應用在傳統同步架構下存在顯著的效能限制。序列化的處理流程包含影格讀取、模型推論與結果輸出，各階段間的依賴關係導致計算資源在等待期間處於閒置狀態。設影片序列 $\tau = \{\tau_1, \tau_2, ..., \tau_n\}$ 包含 $n$ 個連續影格，在傳統同步處理模式下，系統總執行時間可表示為：

$$T_{sync} = \sum_{i=1}^{n} (t_{read,i} + t_{infer,i} + t_{display,i})$$

其中 $t_{read,i}$, $t_{infer,i}$, $t_{display,i}$ 分別代表第 $i$ 個影格的讀取時間、推論時間與顯示時間。同步執行模式要求各階段嚴格依序完成，造成計算單元在非關鍵路徑上的等待，進而限制系統整體處理能力。

為提升系統整體吞吐量並最大化硬體資源使用效率，本專案採用基於 Python asyncio 的非同步多工流水線架構。每個處理階段（Producer/推論/顯示）皆以協程方式獨立運作，並透過佇列（Queue）進行資料傳遞與解耦。

### 架構設計與多工特點

本系統將影像串流處理分為三個主要模組，並支援多個推論 worker 並行運作，worker 數可手動指定（參數 async_workers）：

**模組一：影格讀取（Producer）**
負責從影片檔案或攝影機連續讀取影格，並將 (index, frame, capture_time) 放入 input_queue。當影片結束時，會依 worker 數量送出結束訊號（None）。

```python
async def preprocess(input_queue, video_capture, num_sentinels):
    index = 0
    while True:
        ret, frame = await loop.run_in_executor(None, video_capture.read)
        if not ret:
            break
        await input_queue.put((index, frame, time.time()))
        index += 1
    video_capture.release()
    for _ in range(num_sentinels):
        await input_queue.put(None)
```

**模組二：物件偵測推論（Predict Worker）**
每個 worker 會從 input_queue 取出資料，進行 YOLO 推論，並將結果與相關資訊放入 output_queue。若記憶體用量超過 90%，worker 會自動跳過不啟動。

```python
async def predict_worker(worker_id, input_queue, output_queue, model_path):
    import psutil
    mem = psutil.virtual_memory()
    if mem.percent >= 90:
        print(f"predictor[{worker_id}] 啟動時記憶體已超過 90%，自動跳過此 worker。")
        return
    model = YOLO(model_path)
    while True:
        item = await input_queue.get()
        if item is None:
            await output_queue.put(None)
            break
        index, frame, capture_time = item
        result = await asyncio.to_thread(lambda: model.predict(frame)[0])
        await output_queue.put((index, result, capture_time, worker_id, time.time()))
```

**模組三：結果顯示（Consumer）**
從 output_queue 取得推論結果，繪製 FPS 與 worker 資訊後顯示。收到所有 worker 的結束訊號後結束。

```python
async def postprocess(output_queue, num_sentinels, show_fps=True):
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
        # 顯示 FPS
        if show_fps:
            dt = max(1e-6, time.time() - capture_time)
            fps = 1.0 / dt
            cv2.putText(result, f'Worker{worker_id}  FPS:{fps:.1f}  Frame:{index}',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('YOLO Detection Stream (concurrent)', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
```

目前 worker 數量僅支援手動指定（如 `--async_workers 2`），不支援自動依記憶體調整。各模組間以 FIFO 佇列解耦，最大化硬體效能與吞吐量。完整範例請參考 `ultralytics_demo.py`，或直接執行：

```bash
python ultralytics_demo.py --video_path ./data/serve.mp4 --async_workers 30
```
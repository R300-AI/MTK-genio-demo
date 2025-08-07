
# Ultralytics YOLO 即時串流推論完整教學

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

其中 $t_{read,i}$, $t_{infer,i}$, $t_{display,i}$ 分別代表第 $i$ 個影格的讀取時間、推論時間與顯示時間。同步執行模式要求各階段嚴格依序完成，造成計算單元在非關鍵路徑上的等待，進而限制系統整體處理能力。為提升系統整體吞吐量並最大化硬體資源使用效率，本章節提出基於協程的非同步流水線處理架構。

### 非同步流水線架構設計

本系統將影像串流處理分解為三個獨立運作的異步任務，每個任務可以同時並行執行，不需要等待其他任務完成。這種方式類似於工廠的流水線作業，當第一個影格正在進行物件偵測時，第二個影格已經開始讀取，第三個影格的結果正在顯示，大幅提升整體處理效率。

**模組一：影格讀取**
負責從影片檔案或攝影機連續讀取影格，並透過佇列傳遞給後續的推論模組。當影片結束時會發送結束信號。

```python
async def preprocess(input_queue: asyncio.Queue, cap: cv2.VideoCapture) -> None:
    while cap.isOpened():
        ret, frame = cap.read()
        await input_queue.put(frame if ret else None)
        if not ret:
            break
    cap.release()
```

**模組二：物件偵測推論**
從影格佇列取出影格進行 YOLO 物件偵測，將偵測結果繪製在影像上，然後送到輸出佇列準備顯示。

```python
async def predict(input_queue: asyncio.Queue, output_queue: asyncio.Queue, model) -> None:
    while True:
        frame = await input_queue.get()
        if frame is None: 
            await output_queue.put(None)
            break
        results = model.predict(frame, verbose=False)
        await output_queue.put(results[0].plot())
```

**模組三：結果顯示**
從結果佇列取得已標註的影像並顯示在螢幕上。使用者可以按 'q' 鍵退出程式。

```python
async def postprocess(output_queue: asyncio.Queue) -> None:
    while True:
        result = await output_queue.get()
        if result is None:
            break
        cv2.imshow('YOLO Detection Stream', result)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
```

各處理模組透過先進先出（FIFO）訊息佇列實現解耦通訊，消除模組間的同步等待。完整的非同步串流推論系統已整合在 `ultralytics_demo.py` 中，您可以直接執行以下指令來體驗非同步影像串流推論：

```bash
python ultralytics_demo.py --video_path ./data/serve.mp4
```
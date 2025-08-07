
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

### 背景與動機

物件偵測的即時影片串流處理在傳統同步架構中面臨顯著效能瓶頸。影格擷取、推論計算與結果視覺化的序列性質造成運算資源未充分利用的閒置期間。本研究提出非同步流水線架構以最大化吞吐量與資源效率。

### 問題定義

設 τ = {τ₁, τ₂, ..., τₙ} 代表具有 n 個影格的影片序列。傳統同步處理表現出時間複雜度：

T_sync = Σᵢ₌₁ⁿ (t_read_i + t_infer_i + t_display_i)

其中 t_read_i、t_infer_i 與 t_display_i 分別表示影格擷取、推論與視覺化時間。同步依賴關係造成資源閒置時間，限制整體系統吞吐量。

### 提出的非同步流水線架構

我們引入基於生產者-消費者模式的三階段並發處理框架：

**階段 1（生產者）**：影格擷取與預處理

    ```python
    async def preprocess(input_queue, cap):
        while cap.isOpened():
            ret, frame = cap.read()
            await input_queue.put(frame if ret else None)
            if not ret: break
        cap.release()
    ```
**階段 2（處理）**：YOLO 推論計算  

    ```python
    async def predict(input_queue, output_queue, model):
        while True:
            frame = await input_queue.get()
            if frame is None: 
                await output_queue.put(None); break
            results = model.predict(frame, verbose=False)
            await output_queue.put(results[0].plot())
    ```
**階段 3（消費者）**：結果視覺化與顯示

    ```python
    async def postprocess(output_queue):
        while True:
            result = await output_queue.get()
            if result is None: break
            cv2.imshow('Stream', result)
            if cv2.waitKey(1) == ord('q'): break
        cv2.destroyAllWindows()
    ```

各階段透過 FIFO 佇列進行通訊，實現平行執行並消除同步依賴關係。

### 實作

```python
async def main():
    queues = [asyncio.Queue() for _ in range(2)]
    cap = cv2.VideoCapture("./data/serve.mp4")
    model = YOLO("./models/yolov8n_float32.tflite")

    await asyncio.gather(
        preprocess(queues[0], cap),
        predict(queues[0], queues[1], model),
        postprocess(queues[1])
    )

asyncio.run(main())
```
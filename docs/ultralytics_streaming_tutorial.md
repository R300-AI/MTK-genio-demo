
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

即時物件偵測應用在傳統同步架構下存在顯著的效能限制。序列化的處理流程包含影格讀取、模型推論與結果輸出，各階段間的依賴關係導致計算資源在等待期間處於閒置狀態。為提升系統整體吞吐量並最大化硬體資源使用效率，本章節提出基於協程的非同步流水線處理架構。

### 問題表述

設影片序列 τ = {τ₁, τ₂, ..., τₙ} 包含 n 個連續影格。在傳統同步處理模式下，系統總執行時間可表示為：

T_sync = Σᵢ₌₁ⁿ (t_read_i + t_infer_i + t_display_i)

其中 t_read_i、t_infer_i、t_display_i 分別代表第 i 個影格的讀取時間、推論時間與顯示時間。同步執行模式要求各階段嚴格依序完成，造成計算單元在非關鍵路徑上的等待，進而限制系統整體處理能力。

### 非同步流水線架構設計

本研究採用生產者-消費者並發模式，將處理流程分解為三個獨立運行的協程模組：

**模組一：資料生產階段**
負責影格讀取與初步預處理，將處理後的影格數據置入輸入佇列等待後續處理。

```python
async def preprocess(input_queue, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        await input_queue.put(frame if ret else None)
        if not ret: break
    cap.release()
```

**模組二：推論計算階段**
從輸入佇列提取影格數據，執行 YOLO 物件偵測推論，並將標註結果輸出至結果佇列。

```python
async def predict(input_queue, output_queue, model):
    while True:
        frame = await input_queue.get()
        if frame is None: 
            await output_queue.put(None); break
        results = model.predict(frame, verbose=False)
        await output_queue.put(results[0].plot())
```

**模組三：結果處理階段**
從結果佇列取得推論輸出，執行視覺化渲染與螢幕顯示功能。

```python
async def postprocess(output_queue):
    while True:
        result = await output_queue.get()
        if result is None: break
        cv2.imshow('YOLO Detection Stream', result)
        if cv2.waitKey(1) == ord('q'): break
    cv2.destroyAllWindows()
```

各處理模組透過先進先出（FIFO）訊息佇列實現解耦通訊，消除模組間的同步等待依賴，達成真正的並行處理效果。

### 系統實現

完整的非同步處理系統整合如下：

```python
import asyncio
from ultralytics import YOLO
import cv2

async def main():
    # 建立模組間通訊佇列
    input_queue = asyncio.Queue(maxsize=10)
    output_queue = asyncio.Queue(maxsize=10)
    
    # 初始化系統元件
    video_source = cv2.VideoCapture("./data/serve.mp4")
    detection_model = YOLO("./models/yolov8n_float32.tflite", task='detect')
    
    # 並發執行三個處理模組
    await asyncio.gather(
        preprocess(input_queue, video_source),
        predict(input_queue, output_queue, detection_model),
        postprocess(output_queue)
    )

if __name__ == "__main__":
    asyncio.run(main())
```
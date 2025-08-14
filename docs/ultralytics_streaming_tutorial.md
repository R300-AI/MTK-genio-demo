
# Ultralytics YOLO 即時串流推論教學

本文件將詳細說明如何在 MediaTek Genio 平台上配置並運行 Ultralytics YOLO 的高效能即時串流推論系統。


## 系統配置與前置作業

### 第一步：理解 Ultralytics 推論後端機制

Ultralytics YOLO 框架支援多種推論後端，但預設的 TFLite 解釋器並未針對 MediaTek Genio 平台優化。為了獲得最佳效能，您需要手動配置適合的推論後端。

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
    #     dla_path="<path_to_your_dla_model>",       
    #     device= "<mdla3.0, mdla2.0 or vpu>"
    #     admin_password='<enter_your_admin_password_here>'
    # )
    # LOGGER.info("Successfully loaded NeuronRT delegate for DLA inference")
```

---

### 第二步：手動配置推論後端

請根據您的硬體平台與需求，選擇合適的加速方案，並依下列步驟調整程式碼：

#### ArmNN Delegate（CPU/GPU 加速）

適用於：希望利用 ARM CPU 或 Mali GPU 進行推論加速的情境。

設定步驟：
1. 註解掉原本的解譯器設定（`interpreter = Interpreter(model_path=w)` 和 `raise RuntimeError(...)`）。
2. 取消註解「選項 A」區塊的所有程式碼（移除每行前的 `#`）。
3. 將 `<path to libarmnnDelegate.so>` 替換為您系統上 ArmNN delegate 的實際路徑，例如 `/usr/lib/aarch64-linux-gnu/libarmnnDelegate.so`。
4. 將 `<CpuAcc or GpuAcc>` 設為：
   - `"CpuAcc"`：使用 ARM CPU 加速
   - `"GpuAcc"`：使用 Mali GPU 加速

#### NeuronRT Delegate（MDLA/VPU 加速）

適用於：希望利用 MediaTek 專用的 MDLA 或 VPU 進行推論加速的情境。

設定步驟：
1. 註解掉原本的解譯器設定（`interpreter = Interpreter(model_path=w)` 和 `raise RuntimeError(...)`）。
2. 取消註解「選項 B」區塊的所有程式碼（移除每行前的 `#`）。
3. 將 `<path_to_your_dla_model>` 替換為您的 DLA 模型檔案路徑。
4. 將 `<mdla3.0, mdla2.0 or vpu>` 設為：
   - `"mdla3.0"`：使用 MDLA 3.0
   - `"mdla2.0"`：使用 MDLA 2.0
   - `"vpu"`：使用 VPU
5. 將 `<enter_your_admin_password_here>` 替換為您的系統管理員密碼。

---

### 第三步：驗證推論後端配置

完成上述步驟後，您可以使用以下程式碼驗證配置是否正確：

```python
from ultralytics import YOLO

# 載入模型（會自動使用您配置的後端）
model = YOLO("./models/yolov8n_float32.tflite")

# 執行推論測試
results = model.predict(["./data/bus.jpg"])
results[0].show()

# 如果成功顯示結果，表示後端配置正確
```

---

## 非同步化串流推論架構

即時物件偵測應用在傳統的同步架構下，存在顯著的效能瓶頸。其序列化的處理流程（影格讀取 → 模型推論 → 結果輸出）要求各階段依序完成，導致處理器、加速器等計算資源在等待 I/O 操作時處於閒置狀態，大幅限制了系統的整體吞吐量。

設影片序列 $\tau = \{\tau_1, \tau_2, ..., \tau_n\}$ 包含 $n$ 個連續影格，在傳統同步處理模式下，系統總執行時間可表示為：

$$T_{sync} = \sum_{i=1}^{n} (t_{read,i} + t_{infer,i} + t_{display,i})$$

其中：
- $t_{read,i}$：第 $i$ 個影格的讀取時間
- $t_{infer,i}$：第 $i$ 個影格的推論時間  
- $t_{display,i}$：第 $i$ 個影格的顯示時間


### 多工流水線設計

為了解決上述瓶頸，本教學採用**多工流水線 (Asynchronous Multitasking Pipeline)** 的原理設計此架構。此架構將序列化流程分解為四個獨立且並行運作的組件，透過資料佇列進行解耦與協作，實現高效的推論流程：

* **Producer（生產者）**：作為生產線的起點，負責持續從影片來源讀取原始資料（影格），並將其放入佇列，確保後續處理環節有穩定且充足的資料供應。

    ```python
    while not self.should_stop:
        ret, frame = cap.read()
        await self.frame_queue.put((frame, frame_id))
        # ...記錄統計與結束信號...
    ```

* **Worker（工作者）**：如同產線上的多位並行運作的技術員，從佇列中取得影格進行核心的 AI 推論運算。每位工作者一次只專注處理一個影格，但允許多位工作者同時處理不同影格，藉此最大化推論產能。

    ```python
    while not self.should_stop:
        frame, frame_id = await self.frame_queue.get()
        result = ... # 執行推論
        await self.result_queue.put((result, frame_id, ...))
    ```

* **Manager（工作管理者）**：扮演現場主管的角色，持續監控佇列中待處理的影格數量。當偵測到負載增加（即佇列長度超過閾值）時，能動態增派新的工作者，確保生產線的流暢與高效。

    ```python
    while not self.should_stop:
        avg_queue_length = ... # 計算滑動平均
        if avg_queue_length > 1.0:
            await self._add_worker()
    ```

* **Consumer（消費者）**：位於生產線的終點，負責從佇列中取出已完成推論的結果，並確保多工作者並行下的結果能正確、有序地呈現給使用者。

    ```python
    # 關鍵流程
    while not self.should_stop:
        result, frame_id, ... = await self.result_queue.get()
        # ...依序顯示結果...
    ```
    
---

## 執行範例與參數設定

```bash
python test.py --video_path ./data/video.mp4 \
               --tflite_model ./models/yolov8n_float32.tflite \
               --min_workers 1 \
               --max_workers 6 \
               --queue_size 32
```

**參數說明：**
- `--video_path`: 輸入影片檔案路徑
- `--tflite_model`: TFLite 模型檔案路徑  
- `--min_workers`: 初始工作者數量（建議：1-2）
- `--max_workers`: 最大工作者數量（建議：CPU 核心數）
- `--queue_size`: 影格佇列大小（建議：16-64）

系統會自動將效能統計資料輸出至 `performance_stats.txt`，包含吞吐量、佇列狀態、工作者數量變化等資訊，方便分析系統效能表現。
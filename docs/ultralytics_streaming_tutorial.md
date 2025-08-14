
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

### 第二步：手動配置推論後端

請根據您的硬體平台與需求，選擇合適的加速方案，並依下列步驟調整程式碼：

---
#### ArmNN Delegate（CPU/GPU 加速）

適用於：希望利用 ARM CPU 或 Mali GPU 進行推論加速的情境。

設定步驟：
1. 註解掉原本的解譯器設定（`interpreter = Interpreter(model_path=w)` 和 `raise RuntimeError(...)`）。
2. 取消註解「選項 A」區塊的所有程式碼（移除每行前的 `#`）。
3. 將 `<path to libarmnnDelegate.so>` 替換為您系統上 ArmNN delegate 的實際路徑，例如 `/usr/lib/aarch64-linux-gnu/libarmnnDelegate.so`。
4. 將 `<CpuAcc or GpuAcc>` 設為：
   - `"CpuAcc"`：使用 ARM CPU 加速
   - `"GpuAcc"`：使用 Mali GPU 加速

---

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

即時物件偵測應用在傳統同步架構下存在顯著的效能限制。序列化的處理流程包含影格讀取、模型推論與結果輸出，各階段間的依賴關係導致計算資源在等待期間處於閒置狀態。

設影片序列 $\tau = \{\tau_1, \tau_2, ..., \tau_n\}$ 包含 $n$ 個連續影格，在傳統同步處理模式下，系統總執行時間可表示為：

$$T_{sync} = \sum_{i=1}^{n} (t_{read,i} + t_{infer,i} + t_{display,i})$$

其中：
- $t_{read,i}$：第 $i$ 個影格的讀取時間
- $t_{infer,i}$：第 $i$ 個影格的推論時間  
- $t_{display,i}$：第 $i$ 個影格的顯示時間

為提升系統整體吞吐量並最大化硬體資源使用效率，這項教學將以**多工流水線**作為範例，將序列化處理流程分解為四個獨立且並行運作的組件，讓整個推論流程如同一條高效的自動化生產線，各環節協同合作、無縫接軌：

* **Producer（生產者）**：如同生產線的進料員，負責持續將原料（影格）送入生產線，確保後續每個環節都能獲得穩定且充足的資料來源。
* **Worker（工作者）**：類比於生產線上的多位技術員，負責對每一份原料（影格）進行 AI 推論。每位工作者一次只專注於處理一個影格，處理完畢後才會接手下一份，因此，多位工作者同時並行作業便能有效提升整體產能。
* **Manager（工作管理者）**：就像現場主管，持續監控生產線上待處理原料的堆積狀況，並根據實際負載自動調整技術員（工作者）的人數，確保生產線順暢運作、不會塞車。
* **Consumer（消費者）**：負責將已經加工完成的產品（推論結果）依照正確的順序包裝、展示或送出，最終呈現給使用者。

### 消費者-工作者模式

生產線讓生產者、工作者、消費者透過資料佇列彼此交換資料，讓每個階段都能獨立且高效地運作，實現即時且穩定的推論服務。

**模組一：影格生產者（Frame Producer）**  
負責從影片檔案或攝影機連續讀取影格，並將每個影格（含 frame_id）放入 frame_queue。生產者同時會記錄隊列長度等統計資訊，供管理器參考。當影片結束時，會自動發送結束信號給所有工作者。

```python
# 關鍵流程
while not self.should_stop:
    ret, frame = cap.read()
    await self.frame_queue.put((frame, frame_id))
    # ...記錄統計與結束信號...
```

---

**模組二：智能推論工作者（Inference Worker）**  
多個工作者並行從 frame_queue 取出影格進行 YOLO 推論，並將結果放入 result_queue。每位工作者同時只會處理一個影格，採用「只增不減」策略，啟動後持續運作直到程式結束，並具備錯誤容忍機制。

```python
# 關鍵流程
while not self.should_stop:
    frame, frame_id = await self.frame_queue.get()
    result = ... # 執行推論
    await self.result_queue.put((result, frame_id, ...))
```

---

**模組三：自適應工作者管理器（Workers Manager）**  
持續監控 frame_queue 的積壓情況，採用滑動窗口計算平均隊列長度。當平均值超過設定閾值時，會立即自動增加新的工作者，無冷卻時間限制，確保系統能即時因應負載變化。

```python
# 關鍵流程
while not self.should_stop:
    avg_queue_length = ... # 計算滑動平均
    if avg_queue_length > 1.0:
        await self._add_worker()
```

---

**模組四：結果消費者（Result Consumer）**  
從 result_queue 取得推論結果，並依照 frame_id 順序顯示或輸出。消費者會維護暫存區，確保多工作者並行下的結果能正確、有序地呈現給使用者。

```python
# 關鍵流程
while not self.should_stop:
    result, frame_id, ... = await self.result_queue.get()
    # ...依序顯示結果...
```

### 預載入模型池與性能優化

系統採用預載入模型池技術，在初始化時預先載入最大工作者數量的模型實例，避免運行時的模型載入延遲：

```python
def _preload_models(self):
    """預載入模型池以避免運行時載入延遲"""
    with ThreadPoolExecutor(max_workers=self.max_workers) as preload_executor:
        futures = [preload_executor.submit(load_model, i) for i in range(self.max_workers)]
        for future in futures:
            thread_id, model = future.result()
            self.models[thread_id] = model
```

### 性能監控與日誌記錄

系統內建完整的性能監控機制，所有關鍵指標都會記錄到 `performance_stats.txt` 文件：

- `PROCESSING_TIME`: 每次推論的處理時間
- `QUEUE_LENGTH`: 影格隊列長度變化
- `WORKER_ADDED`: 工作者增加事件
- `MANAGER_STATUS`: 工作者管理器狀態
- `RESULT_RECEIVED`: 結果接收統計

### 使用方式

```bash
python test.py --video_path ./data/video.mp4 \
               --tflite_model ./models/yolov8n_float32.tflite \
               --min_workers 1 \
               --max_workers 6 \
               --queue_size 32
```

**參數說明：**
- `--min_workers`: 初始工作者數量（預設：1）
- `--max_workers`: 最大工作者數量（預設：6）
- `--queue_size`: 影格隊列大小（預設：32）

系統會根據負載自動將工作者數從 `min_workers` 增加到 `max_workers`，實現真正的自適應負載平衡。每次執行時會自動重置 `performance_stats.txt` 文件，開始記錄新的性能統計數據。
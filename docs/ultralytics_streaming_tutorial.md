
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
    #     dla_path="<path_to_your_dla_model>",       
    #     device= "<mdla3.0, mdla2.0 or vpu>"
    #     admin_password='<enter_your_admin_password_here>'
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

### 自適應工作者管理架構

本系統採用智能的自適應工作者管理機制，能夠根據隊列積壓情況自動增加推論工作者，實現真正的動態負載平衡。系統包含四個主要模組：

**模組一：影格生產者（Frame Producer）**
負責從影片檔案或攝影機連續讀取影格，並將 (frame, frame_id) 放入 frame_queue。同時記錄隊列長度統計，供工作者管理器參考。

```python
async def frame_producer(self, cap):
    """生產者：讀取視頻幀並放入隊列"""
    frame_id = 0
    
    while not self.should_stop:
        ret, frame = cap.read()
        if not ret:
            self.should_stop = True
            break
        
        frame_id += 1
        
        try:
            queue_size = self.frame_queue.qsize()
            # 記錄隊列長度
            self.perf_logger.info(f"QUEUE_LENGTH,{frame_id},{queue_size}")
            await self.frame_queue.put((frame, frame_id))
            
        except asyncio.QueueFull:
            self.perf_logger.info(f"QUEUE_FULL,{frame_id}")
            continue
            
        # 減少讓出頻率
        if frame_id % 5 == 0:
            await asyncio.sleep(0.001)
    
    # 發送結束信號
    for _ in range(self.max_workers):
        await self.frame_queue.put(None)
```

**模組二：智能推論工作者（Inference Worker）**
每個工作者從 frame_queue 取出資料，進行 YOLO 推論，並將結果放入 result_queue。工作者採用「只增不減」策略，一旦啟動就會持續運作直到程序結束，避免頻繁創建/銷毀的開銷。

```python
async def inference_worker(self, worker_id):
    """推理工作者 - 移除空閒超時退出邏輯"""
    consecutive_errors = 0
    
    try:
        while not self.should_stop:
            try:
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=5.0)
                
                if frame_data is None:
                    break
                
                frame, frame_id = frame_data
                consecutive_errors = 0
                
                loop = asyncio.get_event_loop()
                result, result_frame_id, processing_time = await loop.run_in_executor(
                    self.executor, self.predict_frame, (frame, frame_id, worker_id)
                )
                
                await self.result_queue.put((result, result_frame_id, frame, processing_time))
                self.frame_queue.task_done()
                
            except asyncio.TimeoutError:
                # 記錄超時但不退出
                self.perf_logger.info(f"WORKER_TIMEOUT,{worker_id}")
                continue
                    
            except Exception as e:
                consecutive_errors += 1
                self.perf_logger.info(f"WORKER_ERROR,{worker_id},{consecutive_errors},{str(e)}")
                if consecutive_errors > 10:
                    break
                continue
                
    finally:
        # 工作者退出時更新計數
        async with self.workers_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                self.current_workers -= 1
```

**模組三：自適應工作者管理器（Workers Manager）**
這是系統的核心創新，負責監控隊列狀態並動態調整工作者數量。採用滑動窗口統計避免瞬間波動的影響，當平均隊列長度超過閾值時立即增加工作者，無冷卻時間限制。

```python
async def workers_manager(self):
    """改進的動態工作者管理器 - 只增不減模式，無冷卻限制"""
    queue_stats = deque(maxlen=5)  # 統計窗口，保留最近5次的隊列長度
    
    while not self.should_stop:
        await asyncio.sleep(0.5)  # 每0.5秒檢查一次
        
        current_queue_size = self.frame_queue.qsize()
        queue_stats.append(current_queue_size)
        
        if len(queue_stats) < 3:  # 至少需要3個統計點
            continue
        
        avg_queue_length = sum(queue_stats) / len(queue_stats)
        
        # 記錄管理器狀態
        self.perf_logger.info(f"MANAGER_STATUS,{self.current_workers},{avg_queue_length:.2f}")
        
        async with self.workers_lock:
            # 只增加工作者的邏輯 - 無冷卻時間限制
            if avg_queue_length > 1.0 and self.current_workers < self.max_workers:
                await self._add_worker()
                self.perf_logger.info(f"WORKER_ADDED,{self.current_workers}")
                print(f"Added worker, current workers: {self.current_workers}")
```

**模組四：結果消費者（Result Consumer）**
從 result_queue 取得推論結果，確保按照影格順序顯示結果。維護一個暫存字典來處理亂序到達的結果，保證顯示的連續性。

```python
async def result_consumer(self):
    """消費者：顯示推理結果"""
    processed_frames = {}
    next_display_frame = 1
    
    while not self.should_stop:
        try:
            result_data = await asyncio.wait_for(self.result_queue.get(), timeout=3.0)
            
            if result_data is None:
                break
            
            result, frame_id, original_frame, processing_time = result_data
            processed_frames[frame_id] = (result, original_frame)
            
            # 記錄結果佇列狀態
            self.perf_logger.info(f"RESULT_RECEIVED,{frame_id},{len(processed_frames)}")
            
            # 按順序顯示結果
            while next_display_frame in processed_frames:
                display_result, display_frame = processed_frames.pop(next_display_frame)
                
                display_img = cv2.resize(display_result.plot(), (720, 480))
                cv2.imshow("Adaptive YOLO Inference", display_img)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    self.should_stop = True
                    return
                
                next_display_frame += 1
            
            self.result_queue.task_done()
            
        except asyncio.TimeoutError:
            if self.frame_queue.empty() and self.result_queue.empty() and self.should_stop:
                break
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
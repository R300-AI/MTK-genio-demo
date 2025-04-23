# Optimizing TFLite Inference Performance with Asynchronous Streaming
## 什麼是Asynchronous

Asynchronous (非同步) 是一種程式設計模型，允許多個任務同時執行，而不需要等待其他任務完成。這種方式特別適合處理 I/O 密集型任務，例如影像處理中的資料讀取、模型推論和結果顯示。以下程式展示了如何使用 Python 的 asyncio 模組來實現非同步的 TFLite 推論。

## 程式分段說明
### 1. 資料預處理 (preprocess)
這個函式負責從影片中讀取影像幀並將其放入輸入佇列 (`input_queue`) 中。
```python
async def preprocess(input_queue, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            await input_queue.put(None)  # 當影片結束時，放入 None 作為結束標記
            break
        await input_queue.put(frame)  # 將每一幀放入輸入佇列
    cap.release()  # 釋放影片資源
```
### 2. 模型推論 (predict)
這個函式負責從輸入佇列中取出影像幀，進行模型推論，並將結果放入輸出佇列 (`output_queue) 中。
```python
async def predict(input_queue, output_queue, model):
    while True:
        frame = await input_queue.get()  # 從輸入佇列中取出影像幀
        if frame is None:
            await output_queue.put(None)  # 當接收到結束標記時，放入 None 作為結束標記
            break
        results = model.predict(frame, verbose=False)  # 使用模型進行推論
        await output_queue.put(results[0].plot())  # 將推論結果放入輸出佇列
```
### 3. 結果後處理 (postprocess)
這個函式負責從輸出佇列中取出推論結果，顯示影像並計算處理速度。
```python
async def postprocess(output_queue):
    while True:
        start_time = time.time()  # 記錄開始時間
        result = await output_queue.get()  # 從輸出佇列中取出推論結果
        if result is None:
            break
        cv2.imshow('streaming', result)  # 顯示推論結果
        print('Streaming Speed:', (time.time() - start_time) * 1000, 'ms')  # 計算處理速度
        if cv2.waitKey(1) == ord('q'):  # 按下 'q' 鍵退出
            break
    cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗
```
### 主程式 (main)
主程式負責初始化佇列、影片資源和模型，並啟動所有非同步任務。
```python
async def main():
    input_queue = asyncio.Queue()  # 初始化輸入佇列
    output_queue = asyncio.Queue()  # 初始化輸出佇列
    cap = cv2.VideoCapture("./data/serve.mp4")  # 打開影片檔案
    model = YOLO("./models/yolov8n_float32.tflite", task='detect')  # 加載 YOLO 模型
    
    await asyncio.gather(
        preprocess(input_queue, cap),  # 啟動資料預處理任務
        predict(input_queue, output_queue, model),  # 啟動模型推論任務
        postprocess(output_queue)  # 啟動結果後處理任務
    )
```

## Run the Asynchronous Inference
```bash
python ultralytics_async.py
```
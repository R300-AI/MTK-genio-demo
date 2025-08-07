# Ultralytics YOLO 與 TFLite 非同步推論整合案例

本案例將結合「Ultralytics YOLO 電腦視覺應用」與「非同步串流最佳化 TFLite 推論效能」兩大主題，展示如何在 MediaTek Genio 平台上，實現高效能的 AI 影像辨識與即時推論。

## 案例目標
- 以 Ultralytics YOLO 模型為基礎，進行物件偵測任務
- 採用 TFLite Runtime 進行模型推論，並結合非同步串流技術提升效能
- 展示如何在 Genio EVK 上串接影像來源、推論、與結果顯示的完整流程

## 環境需求
- MediaTek Genio EVK (510/700/1200)
- Python 3.12
- 已安裝 TFLite Runtime、Ultralytics、OpenCV
- 建議參考本專案 `requirements.txt` 安裝所有依賴

## 主要程式架構
1. **模型準備**：使用 Ultralytics YOLO 預訓練模型，並轉換為 TFLite 格式
2. **資料串流**：以 OpenCV 讀取影像或影片串流
3. **非同步推論**：利用 asyncio 與多執行緒/多進程，實現影像預處理、推論、後處理的流水線
4. **結果顯示**：即時將推論結果疊加於影像並顯示

## 範例流程說明

### 1. 載入 TFLite YOLO 模型
```python
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path='models/yolov8n_float32.tflite')
interpreter.allocate_tensors()
```

### 2. 非同步影像串流與推論
```python
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_inference(frame, interpreter):
    # 影像預處理 ...
    # 推論 ...
    # 後處理 ...
    return result

def read_stream():
    cap = cv2.VideoCapture('data/serve.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

async def main():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        for frame in read_stream():
            result = await loop.run_in_executor(pool, async_inference, frame, interpreter)
            # 顯示結果 ...

asyncio.run(main())
```

### 3. 結果視覺化
```python
# 將推論結果畫在 frame 上
cv2.imshow('YOLO Detection', frame)
cv2.waitKey(1)
```

## 重點說明
- **Ultralytics YOLO**：提供高準確率的物件偵測模型，適合多種場景
- **TFLite Runtime**：輕量化推論引擎，適合嵌入式與邊緣設備
- **非同步/多執行緒**：大幅提升串流推論效能，降低延遲
- **OpenCV 串流**：支援多種影像來源，易於整合

## 延伸應用
- 可結合攝影機即時串流、錄影檔、或多路影像來源
- 可根據硬體資源調整執行緒數量與批次處理策略
- 支援多種 TFLite 模型與自訂後處理

---

如需完整程式碼，請參考本專案 `ultralytics_async.py` 及 `ultralytics_demo.py`，或聯絡專案維護者取得更多教學資源。

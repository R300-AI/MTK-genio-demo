# 基本串流處理Demo

這是一個基本的多進程串流處理示例，展示了Producer-Workers-Consumer模式的實現。

## 功能特色

- **Producer**: 使用OpenCV從視頻文件讀取幀並放入輸入隊列
- **Workers**: 多個並行工作進程從輸入隊列取得幀，進行處理後放入輸出隊列
- **Consumer**: 按照FIFO順序從輸出隊列取得結果並打印

## 架構說明

```
視頻文件 → Producer → Input Queue → Workers → Output Queue → Consumer → 結果輸出
                         ↓
                    [多個並行進程]
```

## 使用方法

### 基本用法
```bash
python basic.py
```

### 自定義參數
```bash
python basic.py --video_path ./data/your_video.mp4 --num_workers 4
```

### 參數說明
- `--video_path`: 輸入視頻文件路徑 (預設: ./data/video.mp4)
- `--num_workers`: Worker進程數量 (預設: 2)

## 測試運行

使用測試腳本來驗證功能：
```bash
python test_streaming.py
```

## 依賴要求

- Python 3.7+
- OpenCV (opencv-python)
- numpy
- multiprocessing (Python內建)

## 程序流程

1. **Producer進程**:
   - 打開視頻文件
   - 逐幀讀取並放入input_queue
   - 控制隊列大小避免記憶體溢出
   - 完成後發送結束信號

2. **Worker進程** (可多個):
   - 從input_queue取得幀數據
   - 調用處理函數 (可替換為實際的AI推理邏輯)
   - 將結果放入output_queue

3. **Consumer進程**:
   - 從output_queue取得處理結果
   - 按照FIFO順序輸出結果
   - 確保結果按幀順序正確顯示

## 自定義處理邏輯

在 `process_frame_subprocess` 函數中，您可以替換為實際的圖像處理邏輯：

```python
def process_frame_subprocess(frame_data):
    frame_id, frame = frame_data
    
    # 在這裡添加您的處理邏輯
    # 例如：物體檢測、圖像分類、特徵提取等
    
    # 返回處理結果
    result = {
        'frame_id': frame_id,
        'your_result': your_processing_result,
        'processed_time': time.time()
    }
    return result
```

## 注意事項

- Windows系統需要使用 `mp.set_start_method('spawn')`
- 隊列大小可根據記憶體情況調整
- Worker數量建議根據CPU核心數和處理複雜度調整
- 支援 Ctrl+C 優雅停止所有進程

## 範例輸出

```
=== 多進程串流處理Demo ===
視頻路徑: ./data/video.mp4
Worker數量: 2
==============================
Producer started: 正在讀取視頻 ./data/video.mp4
Worker 0 started
Worker 1 started
Consumer started
Producer: 已放入第 0 幀
Worker 0: 正在處理第 0 幀
Producer: 已放入第 1 幀
Worker 1: 正在處理第 1 幀
Worker 0: 完成處理第 0 幀
Consumer: 幀 0 結果:
  - 平均顏色 (BGR): [120.5, 98.2, 156.7]
  - 影像尺寸: (480, 640, 3)
  - 處理時間: 14:30:25
--------------------------------------------------
```

# 背壓控制系統優化

## 🎯 解決的問題

原始問題：多 worker 環境下出現批次處理效應
```
時間軸: t0-t4  快速取樣 📸📸📸📸📸
        t5-t15 全部處理中 ⏳⏳⏳⏳⏳  
        t16    突然輸出   💻💻💻💻💻
```

優化後：平滑的輸入輸出流程
```
時間軸: t0  t2  t4  t6  t8  t10 t12 t14
取樣:   📸   📸   📸   📸   📸   📸   📸   📸
處理:    ⚙️   ⚙️   ⚙️   ⚙️   ⚙️   ⚙️   ⚙️
輸出:     💻   💻   💻   💻   💻   💻   💻
```

## 🔧 核心機制

### 1. 背壓控制 (Backpressure Control)
```python
def _should_drop_frame(self):
    busy_workers = sum(1 for worker in self.workers if worker.is_busy)
    queue_size = self.task_queue.qsize() 
    total_load = busy_workers + queue_size
    
    # 當負載超過 80% 時開始丟幀
    return total_load >= (self.max_workers * 0.8)
```

### 2. 均勻派工間隔
```python
# 根據 worker 數量動態調整派工間隔
base_interval = self.balancer.get_producer_sleep(self.process_interval)
self.process_interval = base_interval / max(1, self.max_workers - 2)
```

### 3. 簡化結果處理
- 移除複雜的順序緩衝機制
- 依靠背壓控制自然保證相對順序
- 減少處理延遲

## 🚀 使用方法

### 基本使用
```bash
# 啟用背壓控制（預設）
python gstreamer_demo.py --max_workers 4

# 停用平衡器（不推薦）
python gstreamer_demo.py --max_workers 4 --disable_balancer

# 啟用詳細背壓日誌
python gstreamer_demo.py --max_workers 4 --enable_backpressure_logging
```

### 測試效果
```bash
# 執行自動化測試
python test_backpressure.py
```

## 📊 預期效果

### 改善指標
- ✅ **平衡器調整頻率降低**: 減少 50-80% 的調整次數
- ✅ **處理流程更平滑**: 避免批次處理效應
- ✅ **顯示更穩定**: 減少忽快忽慢的現象
- ✅ **系統負載平衡**: CPU 使用更均勻

### 監控指標
- `Load: X.X%` - 系統負載百分比
- `Backpressure: ON/OFF` - 背壓控制狀態
- `dropping frame to prevent batching` - 背壓丟幀日誌

## 🔍 調試和調優

### 查看系統狀態
```python
# 在程式中可以這樣檢查
worker_pool = WorkerPool(...)
status = worker_pool.get_status()
print(f"系統負載: {status['load_percentage']}%")
print(f"背壓是否啟動: {status['backpressure_active']}")
```

### 調整參數
```python
# 在 _should_drop_frame 方法中調整閾值
load_threshold = self.max_workers * 0.8  # 預設 80%

# 可以根據需要調整為：
load_threshold = self.max_workers * 0.6  # 更保守 (60%)
load_threshold = self.max_workers * 0.9  # 更激進 (90%)
```

## 🎛️ 適用場景

### ✅ 適合的情況
- Worker 數量較少 (2-8 個)
- 處理速度不均勻
- 需要平滑顯示效果
- 實時性要求高

### ⚠️ 需要注意的情況  
- 非常高的 FPS 需求 (可能需要調整丟幀策略)
- 每一幀都很重要的場景 (可以停用背壓控制)
- 處理能力遠超輸入速度時 (背壓控制不會啟動)

## 🐛 故障排除

### 問題：背壓控制沒有啟動
**現象**: 日誌中沒有 "dropping frame" 訊息
**可能原因**: 系統負載不夠高，處理能力足夠
**解決**: 正常現象，代表系統性能良好

### 問題：丟幀太多
**現象**: 大量 "System overloaded" 訊息
**可能原因**: Worker 數量太少或處理太慢
**解決**: 增加 worker 數量或使用更快的模型

### 問題：顯示仍然不平滑
**現象**: 仍有批次處理效應
**可能原因**: 派工間隔設定不當
**解決**: 調整 `self.process_interval` 計算公式

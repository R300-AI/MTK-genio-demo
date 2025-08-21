"""
================================================================================
🖥️ Consumer 架構設計 2025.08.21
================================================================================

Consumer 採用單一職責原則設計，專責推論結果的顯示、輸出與性能監控。
支援 Video（完整性優先）與 Camera（實時性優先）兩種模式，並根據模式自動調整顯示緩衝與丟幀策略。

🎯 核心組件：
┌─────────────┬──────────────────┬─────────────────────────────────────────┐
│ 📸 Video    │ 完整性優先       │ 順序顯示、無丟幀、進度追蹤              │
│ 📷 Camera   │ 實時性優先       │ 智能丟幀、低延遲、FPS自適應            │
└─────────────┴──────────────────┴─────────────────────────────────────────┘

📊 資料流向：
    Results ──> Consumer（顯示/輸出/統計）

🎯 核心架構：
Consumer（顯示與輸出單元）
├── 顯示緩衝管理（根據模式自動調整）
├── FPS與延遲監控
├── 多執行緒顯示循環
└── 統計資料回報

📊 職責分配（◯ = 提供框架 / ✅ = 具體實作）：
┌─────────────────┬───────────────────┬─────────────────┐
│   功能類別      │  Video模式特性      │ Camera模式特性  │
├─────────────────┼───────────────────┼─────────────────┤
│ 🖼️ 顯示緩衝     │ ✅ 順序顯示、無丟幀 │ ✅ 智能丟幀、低延遲 │
│ 🎯 FPS監控      │ ✅ 進度追蹤        │ ✅ FPS自適應      │
│ 🧹 資源管理     │ ✅ 線程安全清理    │ ✅ 線程安全清理    │
│ 📊 統計回報     │ ✅ 顯示統計        │ ✅ 顯示統計        │
└─────────────────┴───────────────────┴──────────────────┘

🔧 核心特性：
• 根據模式自動調整顯示緩衝區大小與丟幀策略
• 支援多執行緒顯示循環，確保主流程不卡頓
• 內建FPS與延遲監控，便於性能分析
• 可與Producer/WorkerPool靈活組合，支援多種串流場景

🛠️ 開發提示：
• 新增顯示模式時，擴充顯示緩衝與丟幀策略分支
• 可根據需求擴充統計資料回報格式
• 注意Video模式下完整性，Camera模式下即時性
"""

import cv2
import time
from collections import deque

# 使用與 pipeline 相同的 logger
 # ...existing code...

import threading

class Consumer:
    def __init__(self, window_name="", monitor=None, display_size=None, fps=30, mode='camera', producer=None):
    # ...existing code...
        self.window_name = window_name
        self.monitor = monitor
        self.display_size = display_size
        self.fps = fps
        self.frame_count = 0
        self.display_times = deque(maxlen=30)
        self.current_fps = 0.0
        self.mode = mode
        self.producer = producer  # 新增：用於獲取原始video fps
        if mode == 'video':
            self.display_buffer = deque(maxlen=50)
        else:
            self.display_buffer = deque(maxlen=1)
        self.display_buffer_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._thread = None

        # 添加fps追蹤變數
        self.display_counter = 0
        self.last_display_fps_time = time.time()
        self.fps_check_interval = 30  # 每30幀檢查一次實際顯示fps
        self.buffer_size_log_counter = 0  # buffer size記錄計數器

        # 初始化日誌
        import logging
        logger = logging.getLogger('gstreamer_demo')
        logger.info("🖥️ " + "="*60)
        logger.info("🖥️ Consumer初始化開始")
        logger.info("🖥️ " + "="*60)
        logger.info(f"📋 步驟 1/2: 🚀 顯示緩衝區配置...")
        logger.info(f"🔍 [{self.mode.upper()}] 視窗名稱: {self.window_name}")
        logger.info(f"🔍 [{self.mode.upper()}] 顯示緩衝區大小: {self.display_buffer.maxlen}")
        logger.info(f"🔍 [{self.mode.upper()}] 目標FPS: {self.fps}")
        logger.info(f"📋 步驟 2/2: ⚙️ 顯示執行緒與統計初始化...")
        logger.info(f"📊 [{self.mode.upper()}] 顯示尺寸: {self.display_size if self.display_size else '原始大小'}")
        logger.info(f"📊 [{self.mode.upper()}] FPS統計間隔: {self.fps_check_interval} 幀")
        logger.info(f"✅ Consumer初始化完成!")
        logger.info("🖥️ " + "="*60)

    def start_display(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._display_loop, daemon=True)
            self._thread.start()

    def stop_display(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def put_frame(self, frame):
        with self.display_buffer_lock:
            if self.mode == 'video':
                # video mode: 確保不丟幀，適當等待 buffer 空間
                while len(self.display_buffer) >= self.display_buffer.maxlen:
                    time.sleep(0.001)  # 短暫等待，避免丟幀
            else:
                # camera mode: 保持原有邏輯，允許丟幀
                if len(self.display_buffer) == self.display_buffer.maxlen:
                    pass
            self.display_buffer.append(frame)

    def _display_loop(self):
        # 動態設定顯示間隔：video mode使用原始fps，camera mode使用設定fps
        if self.mode == 'video' and self.producer and hasattr(self.producer, 'get_fps'):
            target_fps = self.producer.get_fps()
            interval = 1.0 / target_fps
        else:
            interval = 1.0 / self.fps
        
        last_frame = None
        frames_without_new = 0  # 計算連續沒有新幀的次數
        
        while self._running.is_set():
            time.sleep(interval)
            frame = None
            
            with self.display_buffer_lock:
                if self.mode == 'video':
                    # video mode: 依序顯示所有 frame
                    if self.display_buffer:
                        frame = self.display_buffer.popleft()
                        frames_without_new = 0
                    else:
                        frames_without_new += 1
                else:
                    # camera mode: 只顯示最新 frame
                    while self.display_buffer:
                        frame = self.display_buffer.popleft()
                        frames_without_new = 0
            
            # 選擇要顯示的幀
            display_frame = frame if frame is not None else last_frame
            
            # Video mode: 如果連續超過 30 幀(1秒)都沒有新幀，停止重複顯示
            if self.mode == 'video' and frames_without_new > 30:
                display_frame = None
            
            if display_frame is not None:
                self.display_counter += 1
                try:
                    disp_frame = display_frame
                    if self.display_size is not None:
                        disp_frame = cv2.resize(disp_frame, self.display_size)
                    if self.monitor:
                        self.monitor.draw_info(disp_frame)
                    cv2.imshow(self.window_name, disp_frame)
                    
                    # Video mode: 只在顯示新幀時才計數，避免重複計數
                    if self.monitor and (frame is not None or self.mode == 'camera'):
                        self.monitor.count_consumed()
                        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_display()
                        break
                        
                    # 定期記錄顯示fps
                    if self.display_counter % self.fps_check_interval == 0:
                        current_time = time.time()
                        actual_interval = (current_time - self.last_display_fps_time) / self.fps_check_interval
                        actual_display_fps = 1.0 / actual_interval if actual_interval > 0 else 0
                        
                        pass
                        self.last_display_fps_time = current_time
                        
                except Exception as e:
                    pass
            
            # 更新 last_frame 只在有新幀時
            if frame is not None:
                last_frame = frame
        
        # 清理CV2窗口
        cv2.destroyWindow(self.window_name)
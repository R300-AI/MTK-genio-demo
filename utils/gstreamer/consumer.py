"""
================================================================================
🖥️ Consumer 架構設計 2025.08.22
================================================================================

Consumer 採用單一職責原則設計，專責推論結果的顯示、輸出與性能監控。
系統支援 Video（完整性優先）和 Camera（實時性優先）兩種模式差異化處理，並提供統一顯示介面與統計資料，不參與推論執行或結果處理邏輯。。

📊 資料流向：
    YOLO Results ──> Consumer ──> Display Output
                       │
┌───────────────────────────────────────────────────────┐
│ SafeResultHandler ──> result.plot() ──> SimpleBuffer  │
└───────────────────────────────────────────────────────┘
📊 架構組件：
Consumer（主控制器）  
├── SafeResultHandler（核心：安全處理Generator並調用.plot()）  
├── SimpleBuffer（緩衝區：Video/Camera差異化處理）  
└── StatsCollector（統計：FPS監控與回調）

📊 核心功能：
┌─────────────────┬──────────────────────────────┐
│   功能類別      │ 說明內容                      │
├─────────────────┼──────────────────────────────┤
│ ✅ Generator處理 │ 安全提取Generator，帶超時保護 │
│ 🎨 自動渲染     │ 直接調用 result.plot() 方法   │
│ 🖥️ 差異化緩衝   │ Video完整緩衝/Camera實時緩衝  │
│ 📊 統計監控     │ FPS、處理計數、錯誤統計       │
│ 🔧 統一介面     │ start/stop/consume 統一模式   │
└─────────────────┴──────────────────────────────┘

🔧 使用範例：
```python
# 1. 創建配置
config = ConsumerConfig(
    window_name="YOLO Detection",
    display_size=(640, 480),
    mode='video'
)

# 2. 初始化
consumer = Consumer(config)

# 3. 定義統計回調（可選）
def on_stats_update(stats):
    print(f"FPS: {stats['current_fps']:.2f}, 已處理: {stats['total_processed']}")

# 4. 啟動系統
consumer.start(on_stats_update)

# 5. 處理結果（自動調用 result.plot()）
consumer.consume(yolo_result)

# 6. 停止系統
consumer.stop()
```
"""

import cv2
import time
import threading
import logging
import traceback
from collections import deque
from typing import Optional, Callable, Any, Dict

from .config import ConsumerConfig

logger = logging.getLogger('gstreamer_demo')

class SafeResultHandler:
    """🔍 安全結果處理器 - 核心：解決Generator卡住問題並調用.plot()"""
    
    def __init__(self, config: ConsumerConfig):
        self.config = config
        self.processing_count = 0
        self.error_count = 0
        
        logger.info(f"🔍 Safe Handler初始化完成，超時設置: {config.timeout_seconds}s")
    
    def extract_and_plot(self, result: Any) -> Optional[Any]:
        """
        核心方法：安全提取Generator並調用result.plot()
        
        Args:
            result: 來自 WorkerPool 的 YOLO 推論結果
            
        Returns:
            可顯示的幀（調用.plot()的結果），失敗時返回 None
        """
        try:
            logger.info(f"🔍 [SAFE_HANDLER] 處理結果 #{self.processing_count + 1}，類型: {type(result)}")
            
            if result is None:
                logger.warning("⚠️ [SAFE_HANDLER] 收到 None 結果")
                return None
            
            # 🔧 關鍵：安全處理 Generator
            if hasattr(result, '__iter__') and hasattr(result, '__next__'):
                logger.info(f"🔍 [SAFE_HANDLER] 檢測到 Generator，開始安全提取...")
                yolo_results = self._safe_extract_generator(result)
            elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                logger.info(f"🔍 [SAFE_HANDLER] 處理可迭代結果...")
                yolo_results = list(result)
            else:
                logger.info(f"🔍 [SAFE_HANDLER] 處理單一結果...")
                yolo_results = [result]
            
            if not yolo_results:
                logger.warning("⚠️ [SAFE_HANDLER] 空的 YOLO 結果")
                return None
            
            # 🎨 核心：直接調用 .plot() 方法
            yolo_result = yolo_results[0]
            logger.info(f"🎨 [SAFE_HANDLER] 調用 result.plot() 生成顯示幀...")
            
            display_frame = yolo_result.plot(boxes=False)
            
            self.processing_count += 1
            logger.info(f"✅ [SAFE_HANDLER] 成功處理第 {self.processing_count} 個結果")
            
            return display_frame
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ [SAFE_HANDLER] 處理失敗 #{self.error_count}: {e}")
            logger.error(f"❌ 詳細錯誤: {traceback.format_exc()}")
            return None
    
    def _safe_extract_generator(self, generator: Any) -> list:
        """帶超時保護的 Generator 安全提取（Windows 兼容版本）"""
        import threading
        import time
        
        results = []
        extraction_complete = threading.Event()
        extraction_error = None
        
        def extract_worker():
            """在子線程中執行 Generator 提取"""
            nonlocal results, extraction_error
            try:
                logger.info(f"🔍 [SAFE_HANDLER] 開始提取 Generator...")
                
                for i, item in enumerate(generator):
                    results.append(item)
                    logger.info(f"🔍 [SAFE_HANDLER] 已提取第 {i+1} 個項目")
                    
                    # 防止無限循環
                    if i >= 9:
                        logger.warning("⚠️ [SAFE_HANDLER] 達到最大提取限制 (10個項目)")
                        break
                
                logger.info(f"✅ [SAFE_HANDLER] Generator 提取成功: {len(results)} 個結果")
                
            except Exception as e:
                extraction_error = e
                logger.error(f"❌ [SAFE_HANDLER] Generator 提取失敗: {e}")
            finally:
                extraction_complete.set()
        
        # 啟動提取線程
        worker_thread = threading.Thread(target=extract_worker, daemon=True)
        worker_thread.start()
        
        # 等待完成或超時
        if extraction_complete.wait(timeout=self.config.timeout_seconds):
            # 提取完成
            if extraction_error:
                logger.error(f"❌ [SAFE_HANDLER] Generator 提取過程中出錯: {extraction_error}")
                return []
            return results
        else:
            # 超時
            logger.error(f"❌ [SAFE_HANDLER] Generator 提取超時 ({self.config.timeout_seconds}s)")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取處理統計"""
        success_rate = (self.processing_count / (self.processing_count + self.error_count) * 100) if (self.processing_count + self.error_count) > 0 else 0
        return {
            'processing_count': self.processing_count,
            'error_count': self.error_count,
            'success_rate': success_rate
        }

class SimpleBuffer:
    """🖥️ 簡化緩衝區 - Video/Camera 差異化緩衝管理"""
    
    def __init__(self, config: ConsumerConfig):
        self.config = config
        
        # 根據模式設置緩衝區大小
        buffer_size = config.video_buffer_size if config.mode == 'video' else config.camera_buffer_size
        
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.last_frame = None
        self.total_added = 0
        self.dropped_count = 0
        
        logger.info(f"🖥️ Buffer 初始化完成:")
        logger.info(f"   模式: {config.mode}")
        logger.info(f"   緩衝區大小: {buffer_size}")
    
    def put(self, frame: Any) -> bool:
        """添加幀到緩衝區"""
        if frame is None:
            return False
        
        with self.buffer_lock:
            self.total_added += 1
            
            if self.config.mode == 'camera':
                # Camera 模式：只保留最新幀
                self.buffer.clear()
                self.buffer.append(frame.copy())
                logger.debug(f"📦 [BUFFER] Camera模式：更新最新幀")
            else:
                # Video 模式：順序緩衝
                if len(self.buffer) >= self.buffer.maxlen:
                    self.dropped_count += 1
                    logger.debug(f"⚠️ [BUFFER] Video模式：緩衝區滿，丟幀 #{self.dropped_count}")
                
                self.buffer.append(frame.copy())
                logger.debug(f"📦 [BUFFER] Video模式：幀已緩衝 ({len(self.buffer)}/{self.buffer.maxlen})")
            
            self.last_frame = frame.copy()
            return True
    
    def get(self) -> Optional[Any]:
        """從緩衝區獲取幀"""
        with self.buffer_lock:
            if self.buffer:
                frame = self.buffer.popleft()
                logger.debug(f"📤 [BUFFER] 獲取幀 (剩餘: {len(self.buffer)})")
                return frame
            elif self.last_frame is not None and self.config.mode == 'camera':
                # Camera 模式可以重複顯示最後一幀
                logger.debug(f"📤 [BUFFER] Camera模式：返回最後一幀")
                return self.last_frame
            else:
                return None
    
    def clear(self):
        """清空緩衝區"""
        with self.buffer_lock:
            self.buffer.clear()
            logger.info(f"🗑️ [BUFFER] 緩衝區已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取緩衝區統計"""
        with self.buffer_lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.buffer.maxlen,
                'total_added': self.total_added,
                'dropped_count': self.dropped_count,
                'buffer_utilization': len(self.buffer) / self.buffer.maxlen if self.buffer.maxlen > 0 else 0
            }

class StatsCollector:
    """📊 統計收集器 - FPS 監控與回調管理"""
    
    def __init__(self, config: ConsumerConfig, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.config = config
        self.callback = callback
        self.start_time = time.time()
        
        # 統計數據
        self.total_processed = 0
        self.total_displayed = 0
        self.display_times = deque(maxlen=30)  # 用於計算 FPS
        
        # 線程安全
        self.stats_lock = threading.Lock()
        
        logger.info(f"📊 [STATS] 統計收集器初始化，回調間隔: {config.stats_interval}")
    
    def count_processed(self):
        """計數處理的結果"""
        with self.stats_lock:
            self.total_processed += 1
            
            # 定期觸發回調
            if self.total_processed % self.config.stats_interval == 0 and self.callback:
                self._trigger_callback()
    
    def count_displayed(self):
        """計數顯示的幀"""
        with self.stats_lock:
            self.total_displayed += 1
            self.display_times.append(time.time())
    
    def get_current_fps(self) -> float:
        """計算當前 FPS"""
        with self.stats_lock:
            if len(self.display_times) < 2:
                return 0.0
            
            time_span = self.display_times[-1] - self.display_times[0]
            return (len(self.display_times) - 1) / time_span if time_span > 0 else 0.0
    
    def get_current_stats(self) -> Dict[str, Any]:
        """獲取當前統計數據"""
        with self.stats_lock:
            runtime = time.time() - self.start_time
            return {
                'runtime_seconds': runtime,
                'total_processed': self.total_processed,
                'total_displayed': self.total_displayed,
                'current_fps': self.get_current_fps(),
                'average_fps': self.total_displayed / runtime if runtime > 0 else 0
            }
    
    def _trigger_callback(self):
        """觸發統計回調"""
        try:
            stats = self.get_current_stats()
            self.callback(stats)
        except Exception as e:
            logger.error(f"❌ [STATS] 回調執行失敗: {e}")

class Consumer:
    """🖥️ Consumer 主控制器 - 簡化統一管理"""
    
    def __init__(self, **kwargs):
        """
        直接參數初始化方式
        
        Args:
            **kwargs: 直接參數
                - window_name: 視窗名稱
                - display_size: 顯示尺寸 (width, height)
                - fps: 顯示幀率
                - mode: 模式 ('video' 或 'camera')，會自動轉換大小寫
                - timeout_seconds: Generator 提取超時
                - video_buffer_size: Video 模式緩衝區大小
                - camera_buffer_size: Camera 模式緩衝區大小
                - stats_interval: 統計回調間隔
        """
        # 轉換 mode 格式（從 'VIDEO'/'CAMERA' 轉為 'video'/'camera'）
        if 'mode' in kwargs and isinstance(kwargs['mode'], str):
            kwargs['mode'] = kwargs['mode'].lower()
        
        # 從 kwargs 創建配置
        self.config = ConsumerConfig(**kwargs)
        
        logger.info("🏭 " + "="*60)
        logger.info("🏭 Consumer初始化開始")
        logger.info("🏭 " + "="*60)
        
        # 初始化簡化組件
        self.safe_handler = SafeResultHandler(self.config)
        self.buffer = SimpleBuffer(self.config)
        self.stats = None  # 在 start() 時初始化
        
        # 顯示線程管理
        self._running = threading.Event()
        self._display_thread = None
        
        logger.info("✅ Consumer初始化完成!")
    
    def start(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """統一啟動介面"""
        if self._display_thread is not None:
            logger.warning("⚠️ [CONSUMER] 系統已在運行")
            return
        
        # 初始化統計收集器
        self.stats = StatsCollector(self.config, callback)
        
        # 啟動顯示線程
        self._running.set()
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        
        logger.info(f"🚀 [CONSUMER] 系統已啟動")
    
    def stop(self):
        """統一停止介面"""
        if not self._running.is_set():
            return
        
        logger.info(f"🛑 [CONSUMER] 正在停止系統...")
        self._running.clear()
        
        if self._display_thread:
            self._display_thread.join(timeout=3.0)
        
        cv2.destroyWindow(self.config.window_name)
        
        # 最終統計報告
        if self.stats:
            final_stats = self.stats.get_current_stats()
            buffer_stats = self.buffer.get_stats()
            handler_stats = self.safe_handler.get_stats()
            
            logger.info(f"📊 [CONSUMER] 最終統計:")
            logger.info(f"📊   處理結果: {final_stats['total_processed']} 個")
            logger.info(f"📊   顯示幀數: {final_stats['total_displayed']} 幀") 
            logger.info(f"📊   平均FPS: {final_stats['average_fps']:.2f}")
            logger.info(f"📊   處理成功率: {handler_stats['success_rate']:.1f}%")
            logger.info(f"📊   緩衝區利用率: {buffer_stats['buffer_utilization']:.1f}%")
        
        logger.info(f"✅ [CONSUMER] 系統已停止")
    
    def consume(self, result):
        """核心處理方法 - 簡化邏輯"""
        if not self._running.is_set() or not self.stats:
            logger.warning("⚠️ [CONSUMER] 系統未啟動，忽略結果")
            return
        
        logger.info(f"🔄 [CONSUMER] 開始處理結果...")
        
        try:
            # Step 1: 安全提取並調用 .plot()
            display_frame = self.safe_handler.extract_and_plot(result)
            
            if display_frame is None:
                logger.warning("⚠️ [CONSUMER] 結果處理失敗，跳過")
                return
            
            # Step 2: 加入緩衝區
            success = self.buffer.put(display_frame)
            
            if success:
                # Step 3: 更新統計
                self.stats.count_processed()
                logger.info(f"✅ [CONSUMER] 成功處理結果 (總計: {self.stats.total_processed})")
            else:
                logger.warning("⚠️ [CONSUMER] 緩衝區操作失敗")
                
        except Exception as e:
            logger.error(f"❌ [CONSUMER] consume 處理失敗: {e}")
            logger.error(f"❌ 詳細錯誤: {traceback.format_exc()}")
    
    def _display_loop(self):
        """簡化的顯示循環"""
        target_fps = self.config.fps
        frame_interval = 1.0 / target_fps
        
        logger.info(f"🔄 [CONSUMER] 顯示循環開始 (目標FPS: {target_fps})")
        
        while self._running.is_set():
            try:
                frame = self.buffer.get()
                
                if frame is not None:
                    # 調整顯示大小
                    display_frame = frame
                    if self.config.display_size:
                        display_frame = cv2.resize(frame, self.config.display_size)
                    
                    # 顯示幀
                    cv2.imshow(self.config.window_name, display_frame)
                    
                    # 更新顯示統計
                    if self.stats:
                        self.stats.count_displayed()
                    
                    # 檢查退出鍵
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("🔚 [CONSUMER] 用戶按下 'q' 鍵退出")
                        break
                
                # 控制顯示頻率
                time.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"❌ [CONSUMER] 顯示循環錯誤: {e}")
                time.sleep(frame_interval)
        
        logger.info(f"🏁 [CONSUMER] 顯示循環結束")
    
    # 向下兼容方法
    def start_display(self):
        """向下兼容的啟動方法"""
        logger.warning("⚠️ [CONSUMER] start_display() 已廢棄，請使用 start()")
        self.start()
    
    def stop_display(self):
        """向下兼容的停止方法"""
        logger.warning("⚠️ [CONSUMER] stop_display() 已廢棄，請使用 stop()")
        self.stop()
    
    def put_frame(self, frame):
        """向下兼容的 put_frame 方法"""
        logger.warning("⚠️ [CONSUMER] put_frame() 已廢棄，請使用 consume()")
        return self.buffer.put(frame)

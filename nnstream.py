#!/usr/bin/env python3
"""
NNStreamer Python實現 - TensorFlow Lite模型推論
適用於 MediaTek Genio 平台的高效AI推論管道

功能特色:
- 支援 TensorFlow Lite 模型推論
- 異步串流處理架構
- 多工作者並行推論
- 自動資源管理
- 效能監控與最佳化
"""

import asyncio
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import logging
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import psutil
import gc

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nnstreamer.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TFLiteInterpreter:
    """TensorFlow Lite 解釋器封裝類別"""
    
    def __init__(self, model_path: str, num_threads: int = 1):
        self.model_path = model_path
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._initialize_interpreter()
        
    def _initialize_interpreter(self):
        """初始化 TFLite 解釋器"""
        try:
            # 載入模型
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path,
                num_threads=self.num_threads
            )
            self.interpreter.allocate_tensors()
            
            # 獲取輸入輸出詳細資訊
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # 記錄模型資訊
            input_shape = self.input_details[0]['shape']
            logger.info(f"TFLite 模型載入成功: {self.model_path}")
            logger.info(f"輸入張量形狀: {input_shape}")
            logger.info(f"輸出張量數量: {len(self.output_details)}")
            
        except Exception as e:
            logger.error(f"TFLite 模型載入失敗: {e}")
            raise
            
    def get_input_shape(self) -> Tuple[int, ...]:
        """獲取模型輸入形狀"""
        return tuple(self.input_details[0]['shape'])
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """預處理輸入幀"""
        input_shape = self.get_input_shape()
        height, width = input_shape[1], input_shape[2]
        
        # 調整大小
        resized_frame = cv2.resize(frame, (width, height))
        
        # 正規化到 [0, 1]
        if self.input_details[0]['dtype'] == np.float32:
            input_data = resized_frame.astype(np.float32) / 255.0
        else:
            input_data = resized_frame.astype(np.uint8)
            
        # 增加批次維度
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
        
    def inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """執行推論"""
        try:
            # 設定輸入張量
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 執行推論
            self.interpreter.invoke()
            
            # 獲取輸出結果
            outputs = []
            for output_detail in self.output_details:
                output_data = self.interpreter.get_tensor(output_detail['index'])
                outputs.append(output_data)
                
            return outputs
            
        except Exception as e:
            logger.error(f"推論執行失敗: {e}")
            raise


class PerformanceMonitor:
    """效能監控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.fps_times = deque(maxlen=window_size)
        self.last_fps_time = time.time()
        
    def add_inference_time(self, inference_time: float):
        """添加推論時間"""
        self.inference_times.append(inference_time)
        
    def add_fps_sample(self):
        """添加FPS樣本"""
        current_time = time.time()
        if self.fps_times:
            fps = 1.0 / (current_time - self.last_fps_time)
            self.fps_times.append(fps)
        self.last_fps_time = current_time
        
    def get_avg_inference_time(self) -> float:
        """獲取平均推論時間"""
        return np.mean(self.inference_times) if self.inference_times else 0.0
        
    def get_avg_fps(self) -> float:
        """獲取平均FPS"""
        return np.mean(self.fps_times) if self.fps_times else 0.0
        
    def get_system_stats(self) -> Dict[str, Any]:
        """獲取系統統計資訊"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'avg_inference_ms': self.get_avg_inference_time() * 1000,
            'avg_fps': self.get_avg_fps()
        }


class NNStreamer:
    """
    NNStreamer 主要類別
    實現高效的 TensorFlow Lite 模型串流推論
    """
    
    def __init__(self, 
                 model_path: str,
                 input_source: str = None,
                 max_workers: int = 4,
                 queue_size: int = 32,
                 display_output: bool = True):
        
        self.model_path = model_path
        self.input_source = input_source or 0  # 預設使用攝影機
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.display_output = display_output
        
        # 初始化組件
        self.interpreters = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TFLite-Worker")
        self.performance_monitor = PerformanceMonitor()
        
        # 異步隊列
        self.frame_queue = None
        self.result_queue = None
        
        # 控制狀態
        self.should_stop = False
        self.is_running = False
        
        # 初始化系統
        self._initialize_system()
        
    def _initialize_system(self):
        """初始化系統組件"""
        logger.info("=== NNStreamer 系統啟動 ===")
        logger.info(f"TFLite 模型: {self.model_path}")
        logger.info(f"輸入來源: {self.input_source}")
        logger.info(f"最大工作者數: {self.max_workers}")
        
        # 預載入解釋器
        self._preload_interpreters()
        
    def _preload_interpreters(self):
        """預載入 TFLite 解釋器"""
        logger.info("預載入 TFLite 解釋器...")
        
        # 為每個工作者創建獨立的解釋器
        for i in range(self.max_workers):
            thread_id = f"worker_{i}"
            self.interpreters[thread_id] = TFLiteInterpreter(
                model_path=self.model_path,
                num_threads=1  # 每個解釋器使用單執行緒以避免衝突
            )
            
        logger.info(f"成功預載入 {len(self.interpreters)} 個解釋器")
        
    def get_interpreter_for_thread(self) -> TFLiteInterpreter:
        """獲取對應執行緒的解釋器"""
        thread_id = threading.current_thread().name
        
        # 如果當前執行緒沒有對應解釋器，創建新的
        if thread_id not in self.interpreters:
            worker_keys = [k for k in self.interpreters.keys() if k.startswith("worker_")]
            if worker_keys:
                # 重用預載入的解釋器
                available_key = worker_keys[0]
                self.interpreters[thread_id] = self.interpreters.pop(available_key)
            else:
                # 創建新解釋器
                self.interpreters[thread_id] = TFLiteInterpreter(
                    model_path=self.model_path,
                    num_threads=1
                )
                
        return self.interpreters[thread_id]
        
    async def frame_producer(self, source):
        """影像幀生產者協程"""
        logger.info(f"啟動影像擷取器: {source}")
        
        if isinstance(source, str) and source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            logger.error(f"無法開啟影像來源: {source}")
            return
            
        frame_count = 0
        try:
            while not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and source != "0":
                        logger.info("影片播放完畢")
                        break
                    else:
                        continue
                        
                # 將幀放入隊列
                try:
                    await asyncio.wait_for(
                        self.frame_queue.put((frame_count, frame)), 
                        timeout=0.1
                    )
                    frame_count += 1
                except asyncio.TimeoutError:
                    logger.warning("幀隊列已滿，跳過幀")
                    continue
                    
        except Exception as e:
            logger.error(f"影像擷取錯誤: {e}")
        finally:
            cap.release()
            # 發送結束信號
            await self.frame_queue.put(None)
            logger.info("影像擷取器停止")
            
    def sync_inference(self, frame_data: Tuple[int, np.ndarray]) -> Optional[Tuple[int, np.ndarray, List[np.ndarray]]]:
        """同步推論函數（在執行緒中運行）"""
        frame_id, frame = frame_data
        
        try:
            # 獲取解釋器
            interpreter = self.get_interpreter_for_thread()
            
            # 預處理
            start_time = time.time()
            input_data = interpreter.preprocess_frame(frame)
            
            # 執行推論
            outputs = interpreter.inference(input_data)
            
            # 記錄推論時間
            inference_time = time.time() - start_time
            self.performance_monitor.add_inference_time(inference_time)
            
            return frame_id, frame, outputs
            
        except Exception as e:
            logger.error(f"推論失敗 (Frame {frame_id}): {e}")
            return None
            
    async def inference_manager(self):
        """推論管理器協程"""
        logger.info("啟動推論管理器")
        
        active_tasks = set()
        
        try:
            while not self.should_stop:
                # 從幀隊列獲取數據
                frame_data = await self.frame_queue.get()
                
                if frame_data is None:
                    logger.info("收到結束信號")
                    break
                    
                # 創建推論任務
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(
                    self.executor, 
                    self.sync_inference, 
                    frame_data
                )
                active_tasks.add(task)
                
                # 處理完成的任務
                done_tasks = [t for t in active_tasks if t.done()]
                for task in done_tasks:
                    active_tasks.remove(task)
                    result = await task
                    
                    if result is not None:
                        try:
                            await asyncio.wait_for(
                                self.result_queue.put(result),
                                timeout=0.1
                            )
                        except asyncio.TimeoutError:
                            logger.warning("結果隊列已滿")
                            
                # 限制同時進行的任務數量
                while len(active_tasks) >= self.max_workers:
                    done, active_tasks = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in done:
                        result = await task
                        if result is not None:
                            try:
                                await asyncio.wait_for(
                                    self.result_queue.put(result),
                                    timeout=0.1
                                )
                            except asyncio.TimeoutError:
                                logger.warning("結果隊列已滿")
                                
        except Exception as e:
            logger.error(f"推論管理器錯誤: {e}")
        finally:
            # 等待所有任務完成
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            logger.info("推論管理器停止")
            
    def postprocess_results(self, outputs: List[np.ndarray]) -> Dict[str, Any]:
        """後處理推論結果"""
        # 這裡可以根據具體模型進行自定義後處理
        # 以下是通用的結果處理示例
        
        result = {
            'num_outputs': len(outputs),
            'output_shapes': [output.shape for output in outputs],
            'raw_outputs': outputs
        }
        
        # 如果是YOLO類型的檢測模型，可以添加特定處理
        if len(outputs) >= 1:
            # 假設第一個輸出是檢測結果
            detections = outputs[0]
            if detections.ndim >= 2:
                result['num_detections'] = detections.shape[1] if detections.ndim > 1 else 1
                
        return result
        
    async def result_consumer(self):
        """結果消費者協程"""
        logger.info("啟動結果處理器")
        
        if self.display_output:
            cv2.namedWindow("NNStreamer Output", cv2.WINDOW_AUTOSIZE)
            
        try:
            while not self.should_stop:
                try:
                    # 獲取推論結果
                    result = await asyncio.wait_for(
                        self.result_queue.get(),
                        timeout=1.0
                    )
                    
                    if result is None:
                        break
                        
                    frame_id, frame, outputs = result
                    
                    # 後處理結果
                    processed_result = self.postprocess_results(outputs)
                    
                    # 更新效能監控
                    self.performance_monitor.add_fps_sample()
                    
                    # 在幀上繪製結果資訊
                    display_frame = frame.copy()
                    stats = self.performance_monitor.get_system_stats()
                    
                    # 顯示統計資訊
                    info_text = [
                        f"Frame: {frame_id}",
                        f"FPS: {stats['avg_fps']:.1f}",
                        f"Inference: {stats['avg_inference_ms']:.1f}ms",
                        f"CPU: {stats['cpu_percent']:.1f}%",
                        f"Memory: {stats['memory_percent']:.1f}%",
                        f"Outputs: {processed_result['num_outputs']}"
                    ]
                    
                    for i, text in enumerate(info_text):
                        cv2.putText(display_frame, text, (10, 30 + i * 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                    # 顯示輸出形狀資訊
                    if 'output_shapes' in processed_result:
                        for i, shape in enumerate(processed_result['output_shapes']):
                            shape_text = f"Output{i}: {shape}"
                            cv2.putText(display_frame, shape_text, (10, 200 + i * 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # 顯示影像
                    if self.display_output:
                        cv2.imshow("NNStreamer Output", display_frame)
                        
                        # 檢查按鍵退出
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' 或 ESC
                            logger.info("使用者請求退出")
                            self.should_stop = True
                            break
                            
                    # 定期記錄統計資訊
                    if frame_id % 30 == 0:  # 每30幀記錄一次
                        logger.info(f"統計 - FPS: {stats['avg_fps']:.1f}, "
                                  f"推論時間: {stats['avg_inference_ms']:.1f}ms, "
                                  f"CPU: {stats['cpu_percent']:.1f}%, "
                                  f"記憶體: {stats['memory_percent']:.1f}%")
                        
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            logger.error(f"結果處理器錯誤: {e}")
        finally:
            if self.display_output:
                cv2.destroyAllWindows()
            logger.info("結果處理器停止")
            
    async def run(self):
        """運行 NNStreamer 主循環"""
        if self.is_running:
            logger.warning("NNStreamer 已在運行中")
            return
            
        self.is_running = True
        self.should_stop = False
        
        # 初始化異步隊列
        self.frame_queue = asyncio.Queue(maxsize=self.queue_size)
        self.result_queue = asyncio.Queue(maxsize=self.queue_size)
        
        logger.info("=== 啟動 NNStreamer 串流處理 ===")
        
        try:
            # 創建所有協程任務
            tasks = [
                asyncio.create_task(self.frame_producer(self.input_source)),
                asyncio.create_task(self.inference_manager()),
                asyncio.create_task(self.result_consumer())
            ]
            
            # 並行運行所有任務
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("收到中斷信號")
        except Exception as e:
            logger.error(f"運行時錯誤: {e}")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """清理資源"""
        logger.info("清理系統資源...")
        self.should_stop = True
        self.is_running = False
        
        # 關閉執行器
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            
        # 清理解釋器
        self.interpreters.clear()
        
        # 強制垃圾回收
        gc.collect()
        
        logger.info("資源清理完成")
        

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="NNStreamer - TensorFlow Lite 串流推論系統")
    parser.add_argument("--model", type=str, 
                       default="./models/yolov8n_float32.tflite",
                       help="TFLite 模型路徑")
    parser.add_argument("--input", type=str, default="0",
                       help="輸入來源 (攝影機編號或影片路徑)")
    parser.add_argument("--workers", type=int, default=4,
                       help="最大工作者數量")
    parser.add_argument("--queue_size", type=int, default=32,
                       help="隊列大小")
    parser.add_argument("--no_display", action="store_true",
                       help="禁用視覺輸出")
    
    args = parser.parse_args()
    
    # 檢查模型檔案
    if not os.path.exists(args.model):
        logger.error(f"模型檔案不存在: {args.model}")
        return
        
    # 創建 NNStreamer 實例
    nnstreamer = NNStreamer(
        model_path=args.model,
        input_source=args.input,
        max_workers=args.workers,
        queue_size=args.queue_size,
        display_output=not args.no_display
    )
    
    # 運行系統
    try:
        asyncio.run(nnstreamer.run())
    except KeyboardInterrupt:
        logger.info("程式被使用者中斷")
    except Exception as e:
        logger.error(f"程式執行失敗: {e}")
    finally:
        logger.info("程式結束")


if __name__ == "__main__":
    main()

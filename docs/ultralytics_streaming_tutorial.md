# Ultralytics YOLO 即時串流推論教學

本教學將指導您如何在 MediaTek Genio 平台上配置並運行 Ultralytics YOLO 的高效能串流推論系統。

### 第一步：理解Ultralytics 執行模型推論的運算機制

開啟 `ultralytics/nn/autobackend.py`，在 TFLite 區段（約第 465-490 行）您會看到以下程式碼：

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
    #     dla_path="<path to your dla model>",       
    #     device= "<mdla3.0, mdla2.0 or vpu>"
    # )
    # LOGGER.info("Successfully loaded NeuronRT delegate for DLA inference")
```

### 第二步：手動配置推論後端（修改 autobackend.py，選擇 ArmNN 或 NeuronRT）

根據您的硬體與需求，選擇下方其中一種加速方式，將對應區塊的程式碼取消註解，並填入正確參數：

**選項 A：使用 ArmNN delegate（CPU/GPU 加速**

- 取消註解 ArmNN 相關程式碼。
- 將 `library` 參數改為您系統上 `libarmnnDelegate.so` 的實際路徑。
- `options` 的 `backends` 可設為 `"CpuAcc"` 或 `"GpuAcc"`，依照您的硬體選擇。

**選項 B：使用 NeuronRT delegate（MDLA/VPU 加速）**

- 取消註解 NeuronRT 相關程式碼。
- 將 `dla_path` 改為您的 DLA model 路徑。
- `device` 參數請依照您的硬體設為 `"mdla3.0"`、`"mdla2.0"` 或 `"vpu"`。

請務必將「原生 TFLite 解譯器」的（`interpreter = Interpreter(model_path=w)`）以及 `raise RuntimeError(...)` 這註解掉，避免重複執行或出現錯誤。

#### 驗證配置

完成上述步驟後，Ultralytics 就會使用您選擇的推論後端進行加速推論。

```python
from ultralytics import YOLO

model = YOLO("./models/yolov8n_float32.tflite")
results = model.predict(["./data/bus.jpg"])
results[0].show()
```

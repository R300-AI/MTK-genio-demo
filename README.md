# Genio EVK 最佳實作指南


## AI 部署工作流程概述

隨著人工智慧（AI）技術的快速發展，深度學習模型已廣泛應用於影像辨識、語音處理、智慧監控等多種場域。傳統上，AI模型的開發與訓練大多在資源充足的雲端平台或高效能個人電腦上進行，開發者可專注於提升模型的準確度與泛化能力。然而，將這些模型實際應用於終端設備（如邊緣裝置、嵌入式系統）時，往往會面臨運算資源有限、功耗受限、即時性要求高等挑戰。因此，AI部署（AI Deployment）成為連結模型開發與實際應用的關鍵橋樑。

AI部署的核心目標，是將已訓練完成的模型最佳化並移植到目標硬體平台，使其能在受限的運算與記憶體資源下，依然維持高效且穩定的推論效能。這一過程不僅涉及模型格式的轉換（如將TensorFlow/PyTorch模型轉為TFLite、ONNX等輕量格式），還需根據硬體特性進行量化、剪枝、加速器適配等最佳化處理。部署的成效將直接影響AI應用的即時性、可靠性與能源效率。

### 工作流程架構

在MediaTek Genio平台上，AI部署更需考慮多種異質運算資源（如MDLA、Mali GPU、VPU、ARM CPU）的協同運作，並根據應用場景選擇最合適的推論路徑。開發者必須在模型準確度、推論延遲、功耗、相容性等多重目標間取得平衡，才能發揮硬體平台的最大效益。

```
開發階段          →    轉換階段         →    部署階段         →    最佳化階段
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 模型訓練     │   │ 格式轉換      │   │ 硬體驅動安裝  │   │ 效能調校      │
│ TF/PyTorch  │   │ TFLite       │   │ 加速器設定    │   │ 記憶體優化    │
│ 模型驗證     │───→│ 量化壓縮      │───→│ 推論測試      │───→│ 延遲最佳化    │
│ 準確度測試   │   │ DLA編譯      │   │ 應用整合      │   │ 能耗管理      │
└─────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```


本文件將以系統化流程，說明如何在Genio EVK上完成AI模型的部署，並針對部署過程中的關鍵考量與實務細節，提供完整的技術指引，協助開發者順利將AI創新落地於各類智慧終端。

## Genio 處理器架構與模型部署選擇

MediaTek Genio 系列採用先進的異質整合封裝技術，透過小晶片（chiplet）互聯，將 CPU、GPU、MDLA、VPU 等多種運算處理器高效整合於單一晶片內。與雲端平台的單一加速器架構不同，Genio 提供了多元化的運算資源選擇，開發者必須理解這種異質運算架構的特性，才能針對不同 AI 模型的特性與效能需求，選擇最適合的部署策略。

<div align="center">
<img src="https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/images/UCIe-diagram.jpg" width="360"/>
</div>

### 記憶體共享機制

傳統 SoC 架構的主要挑戰在於記憶體頻寬的限制。當 AI 模型需要從 CPU 轉交給 GPU 執行時，必須透過 PCIe 匯流排進行資料傳輸，這個過程包含多次記憶體拷貝：首先將模型與輸入資料從主記憶體複製到 GPU 的專用記憶體，運算完成後再將結果拷貝回來。每一次拷貝都會產生延遲並消耗額外功耗，成為了目前即時 AI 應用的效能瓶頸。

而Genio 平台透過 UCIe 小晶片互聯技術從根本上解決了這個問題。MDLA 與 VPU 等專用 AI 加速器能直接與 CPU 共享主記憶體，實現以下關鍵優勢：

- **直接記憶體存取**：MDLA 和 VPU 可直接讀寫 CPU 主記憶體，無需透過額外的記憶體控制器。模型使用 NeuronRT 將 .tflite 編譯為 .dla 格式後，載入至 MDLA 時無需額外記憶體拷貝，顯著縮短模型載入時間。
- **零複製傳輸**：模型權重和推論資料在 CPU 與 AI 加速器間傳遞時無需重複複製，CPU 可直接存取 MDLA/VPU 的運算結果，實現 pipeline 並行處理，提升推論效率並降低延遲。
- **統一位址空間**：AI 加速器與 CPU 使用相同的記憶體位址映射，簡化軟體開發複雜度。同一份模型資料可被 CPU、MDLA、VPU 共享使用，降低總體記憶體需求並優化記憶體管理。

這種架構設計為開發者帶來更短的模型載入時間、更低的推論延遲、更少的記憶體佔用，在處理高解析度影像或複雜 AI 工作流時優勢更加明顯。MediaTek Genio 系列則針對不同應用場景推出多款AI SoC，型號的運算單元配置如下：

| 型號         | CPU                                 | GPU                | MDLA (DLA)         | VPU         |
|--------------|-------------------------------------|--------------------|--------------------|-------------|
| **Genio 510**| 4x Arm Cortex-A73 + 4x Cortex-A53   | Arm Mali-G57 MC2   | 1x MDLA v3.0       | 1x VPU      |
| **Genio 700**| 2x Arm Cortex-A78 + 6x Cortex-A55   | Arm Mali-G57 MC3   | 1x MDLA v3.0       | 1x VPU      | 
| **Genio 1200**| 4x Arm Cortex-A78 + 4x Cortex-A55  | Arm Mali-G57 MC5   | 2x MDLA v2.0       | 2x VPU      | 





## 事前準備

本專案提供了完整的 AI 模型部署示範環境。在開始之前，請確保您已完成以下準備工作：

1. **硬體設備**：MediaTek Genio EVK 開發板（支援型號：510/700/1200）
2. **系統安裝**：透過 [Getting Start 指南](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/getting_start_with_ubuntu_zh.md) 燒錄 Ubuntu 作業系統，並安裝BSPs（`ArmNN`、`NeuronRT`）
3. **環境設定**：開機並依照以下步驟安裝MTK-genio-demo
    ```bash
    # 安裝 Python 套件管理工具（需使用 Python 3.12）
    curl -LsSf https://astral.sh/uv/install.sh | sh  
    ```
    ```bash
    # 下載專案並安裝相依套件
    $ git clone https://github.com/R300-AI/MTK-genio-demo.git
    $ cd MTK-genio-demo

    $ uv add -r requirements.txt  
    ```

## 快速開始

### 效能測試工具

完成環境設定後，您可以透過Jupyter Lab了解如何在 Genio 平台上進行 AI 模型的推論與效能評估。

```bash
$ uv run --with jupyter jupyter lab
```

這些 Notebook 範例需要您事先準備好 TensorFlow Lite 格式（.tflite）的 AI 模型檔案。本專案已在 `./models/` 目錄中預先提供範例模型(`./models/yolov8n_float32.tflite`)，開發者可直接使用這些模型進行測試。

> - **[ArmNN 模型推論教學](./notebook/armnn_benchmark.ipynb)** - 使用 ArmNN 推論引擎在Arm的 CPU 和 GPU 上執行模型推論與效能評估
> - **[NeuronRT 模型推論教學](./notebook/neuronrt_benchmark.ipynb)** - 使用 NeuronRT 推論引擎在MTK加速器（DLA/VPU）上執行模型推論與效能評估 

### 模型測試工具

如果開發者想針對 MTK 加速器（DLA/VPU）進行模型設計，在實際進行部署前可以預先透過 [**NeuronPilot AI Porting Platform**](https://neuronpilot-porting-platform.azurewebsites.net/) 線上平台驗證模型的相容性。這個平台提供多種模型格式的支援：對於 PyTorch 和 TensorFlow 模型，可透過程式碼執行來檢測模型對各處理器的支援性；對於已經訓練完成的 ONNX 和 TensorFlow Lite 模型，則可直接上傳模型檔以進行相容性驗證及模型編譯/下載。

透過此平台的驗證機制，開發者能夠在設計階段的早期就確認模型架構是否適合 MediaTek 的專用 AI 加速器，避免在後續部署過程中遇到相容性問題，有效提升開發效率。

#### NeuronPilot AI Porting Platform

**NeuronPilot Porting Platform** 是工研院提供的線上模型轉換平台，專門用於將 TensorFlow Lite 模型編譯為 DLA 格式，以便在 MediaTek Genio 系列的 MDLA 加速器上運行。

🌐 **平台網址**：[https://neuronpilot-porting-platform.azurewebsites.net/](https://neuronpilot-porting-platform.azurewebsites.net/)

**使用流程**：
1. **Upload Prebuilt Model** - 上傳您的 TFLite 模型檔案
2. **選擇 TFLite 模型** - 確認模型格式和參數
3. **Upload and Verify Model** - 驗證模型的相容性
4. **選擇開發板與 Device** - 選擇對應的 Genio 型號和加速器類型
5. **Download DLA** - 下載轉換完成的 DLA 格式模型

> ⚠️ **重要提醒**
> - 轉換過程需要網路連線
> - 請確認選擇正確的開發板型號（G510/700 使用 mdla3.0，G1200 使用 mdla2.0）
> - 轉換完成的 DLA 檔案建議存放在 `./models/` 目錄下

完成模型轉換與相容性驗證後，建議參考上述的


## 進階教學

透過以下 Python 範例，您將全面了解在這些晶片上部署 AI 模型的基礎知識，包括應用程式部署、加速模型推論和資料串流。這將幫助您快速掌握 Genio 晶片的全部潛能。建議從與您使用案例相似的功能開始探索：

### 教學清單

* **[使用 Ultralytics YOLO 部署電腦視覺應用](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/demo/ultralytics_tutorial.md)**
* **[使用 Ollama 和 OpenAI 函式庫部署語言處理應用](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/demo/ollama_and_openai_whisper_tutorial.md)**
* **[透過非同步串流最佳化 TFLite 推論效能](https://github.com/R300-AI/MTK-genio-demo/blob/main/docs/demo/async_streaming_tutorial.md)**
* **手動將 TFLite 模型編譯為 DLA 格式的方法 (待完成)**
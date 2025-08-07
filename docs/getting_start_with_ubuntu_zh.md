# Genio EVK 開發環境設定指南
##### 更新時間：2025年01月 by ITRI (EOSL-R3)

## AI 部署工作流程概觀

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
  <img src="../images/UCIe-diagram.jpg" width="480" alt="UCIe異質整合封裝示意圖"/>
</div>

### 記憶體共享機制

傳統 SoC 架構的主要挑戰在於記憶體頻寬的限制。當 AI 模型需要從 CPU 轉交給 GPU 執行時，必須透過 PCIe 匯流排進行資料傳輸，這個過程包含多次記憶體拷貝：首先將模型與輸入資料從主記憶體複製到 GPU 的專用記憶體，運算完成後再將結果拷貝回來。每一次拷貝都會產生延遲並消耗額外功耗，成為即時 AI 應用的效能瓶頸。

Genio 平台透過 UCIe 小晶片互聯技術從根本上解決了這個問題。MDLA 與 VPU 等專用 AI 加速器能直接與 CPU 共享主記憶體，實現以下關鍵優勢：

- **直接記憶體存取**：MDLA 和 VPU 可直接讀寫 CPU 主記憶體，無需透過額外的記憶體控制器。模型使用 NeuronRT 將 .tflite 編譯為 .dla 格式後，載入至 MDLA 時無需額外記憶體拷貝，顯著縮短模型載入時間。
- **零複製傳輸**：模型權重和推論資料在 CPU 與 AI 加速器間傳遞時無需重複複製，CPU 可直接存取 MDLA/VPU 的運算結果，實現 pipeline 並行處理，提升推論效率並降低延遲。
- **統一位址空間**：AI 加速器與 CPU 使用相同的記憶體位址映射，簡化軟體開發複雜度。同一份模型資料可被 CPU、MDLA、VPU 共享使用，降低總體記憶體需求並優化記憶體管理。

這種架構設計為開發者帶來更短的模型載入時間、更低的推論延遲、更少的記憶體佔用，在處理高解析度影像或複雜 AI 工作流時優勢更加明顯。相較之下，傳統 CPU 和 GPU 採用快取階層架構，執行 AI 推論時需要複雜的記憶體管理機制，導致較長的模型載入時間和加速器切換延遲。

MediaTek Genio 系列針對不同應用場景推出多款高效能 AI SoC，各型號的異質運算單元配置如下：

| 型號         | CPU                                 | GPU                | MDLA (DLA)         | VPU         |
|--------------|-------------------------------------|--------------------|--------------------|-------------|
| **Genio 510**| 4x Arm Cortex-A73 + 4x Cortex-A53   | Arm Mali-G57 MC2   | 1x MDLA v3.0       | 1x VPU      |
| **Genio 700**| 2x Arm Cortex-A78 + 6x Cortex-A55   | Arm Mali-G57 MC3   | 1x MDLA v3.0       | 1x VPU      | 
| **Genio 1200**| 4x Arm Cortex-A78 + 4x Cortex-A55  | Arm Mali-G57 MC5   | 2x MDLA v2.0       | 2x VPU      | 


### 先決條件

在開始部署流程之前，請確保您已具備：

- **硬體**: MediaTek Genio EVK (350/510/700/1200)
- **作業系統**: 專用 Ubuntu 映像檔
- **開發經驗**: 基礎 Linux 操作和 Python 程式設計
- **AI 模型**: TensorFlow 或 PyTorch 訓練完成的模型

## 💻 Ubuntu 作業系統安裝

**重要提醒**：與雲端平台不同，您需要先為開發板安裝作業系統。

### 為什麼需要安裝Ubuntu？
在雲端平台上，您直接使用預先配置好的環境。但在嵌入式開發板上，您需要：
1. 安裝適合ARM架構的Ubuntu系統
2. 安裝專用的AI加速驅動程式
3. 配置Python環境和相關套件

### 安裝步驟：

1. **下載Ubuntu映像檔**：
   - 前往 [https://ubuntu.com/download/mediatek-genio](https://ubuntu.com/download/mediatek-genio)
   - 選擇對應您開發板的版本（Genio-350/510/700/1200）
   - 💡 **提醒**：這是專為Genio優化的Ubuntu版本，不是一般的Ubuntu Desktop

2. **燒錄系統**：
   - 按照官方教學 [https://mediatek.gitlab.io/genio/doc/ubuntu/](https://mediatek.gitlab.io/genio/doc/ubuntu/) 
   - 使用燒錄工具將系統安裝到開發板
   - ⚠️ **注意**：這個過程會清除開發板上的所有資料

> ### ✅ 開始前的檢查清單
> * ✅ 一塊已成功燒錄Ubuntu的 **Genio-350、510、700** 或 **1200 EVK** 開發板
> * ✅ 網路連線已設定完成（WiFi6或5G模組連接天線並啟用）
> * ✅ 熟悉基本Linux命令列操作（類似在Colab中使用`!`執行系統命令）
> * ✅ 建議安裝 **Miniconda** 來管理Python環境（就像在本地開發時一樣）


## 🚀 AI加速套件安裝

接下來我們要安裝三個重要的AI加速套件。想像這些套件就像是**不同的深度學習框架**：

| 套件名稱 | 作用類比 | 主要功能 |
|---------|---------|---------|
| **NeuronRT** | 類似TensorRT | 使用MDLA/VPU進行最高效能的AI推論 |
| **ArmNN** | 類似OpenVINO | 在ARM CPU/GPU上優化推論效能 |
| **KleidiAI** | 類似優化版NumPy | 提供ARM平台的高效數學運算 |

---

## 📦 1. NeuronRT：專用AI加速器驅動

**什麼是NeuronRT？**
如果您使用過NVIDIA的TensorRT來加速GPU推論，NeuronRT就是MediaTek的等價物。它讓您的TensorFlow Lite模型能使用MDLA和VPU硬體加速器。

### 🔨 步驟1：安裝CMake（編譯工具）

**為什麼需要CMake？**
與雲端環境不同，邊緣設備經常需要編譯原始碼。CMake是一個跨平台的編譯工具。

```bash
# 安裝SSL開發套件（用於安全連線）
$ sudo apt-get install libssl-dev

# 下載並編譯最新版CMake
$ git clone https://github.com/Kitware/CMake.git && cd CMake
$ ./bootstrap && make && sudo make install && cd ~
```

💡 **解說**：這個過程需要15-30分鐘，類似在本地電腦編譯大型專案。

### 📚 步驟2：安裝NeuronRT函式庫

**重要概念**：NeuronRT包含硬體驅動程式和Python API，讓您能：
- 將`.tflite`模型轉換為`.dla`格式（DLA = Deep Learning Accelerator格式）
- 在MDLA上執行高速推論
- 存取VPU進行電腦視覺加速

```bash
# 按照MediaTek官方指南安裝
```
請遵循[MediaTek官方安裝指南](https://mediatek.gitlab.io/genio/doc/ubuntu/bsp-installation/neuropilot.html#)完成安裝。

**安裝完成後測試**：
```bash
# 檢查是否能正常匯入
$ python3 -c "import neuronrt; print('NeuronRT安裝成功！')"
```

---

## ⚡ 2. ArmNN：ARM平台AI推論加速

**什麼是ArmNN？**
想像ArmNN就像是專為ARM處理器設計的**TensorFlow Lite運行環境**。當MDLA忙碌或不可用時，它能在ARM CPU和Mali GPU上提供優化的AI推論。

### 📥 步驟1：下載預編譯的ArmNN函式庫

**為什麼使用預編譯版本？**
編譯ArmNN需要很長時間和大量記憶體，使用預編譯版本可以節省時間。

```bash
# 在根目錄建立ArmNN專用資料夾
$ sudo mkdir -p /ArmNN-linux-aarch64

# 下載並解壓縮到指定目錄
$ curl -L https://github.com/ARM-software/armnn/releases/download/v24.11/ArmNN-linux-aarch64.tar.gz | sudo tar -xz -C /ArmNN-linux-aarch64 --strip-components=1
```

💡 **解說**：`--strip-components=1` 會移除壓縮檔的頂層資料夾，直接解壓縮內容。

### 🔧 步驟2：設定系統環境變數

**為什麼需要設定環境變數？**
就像在Python中設定`PYTHONPATH`一樣，我們需要告訴系統在哪裡找到ArmNN函式庫。

```bash
# 編輯bashrc檔案（類似設定Python虛擬環境）
$ vim ~/.bashrc

# 在檔案末尾加入這行：
export LD_LIBRARY_PATH=/ArmNN-linux-aarch64:$LD_LIBRARY_PATH

# 重新載入設定
$ source ~/.bashrc
```

**驗證安裝**：
```bash
# 檢查環境變數
$ echo $LD_LIBRARY_PATH
# 應該看到 /ArmNN-linux-aarch64 在路徑中
```

---

## 🧮 3. KleidiAI：ARM平台數學運算優化

**什麼是KleidiAI？**
如果您熟悉Intel的MKL（Math Kernel Library）或NVIDIA的cuBLAS，KleidiAI就是ARM版本的高效數學運算庫。它優化了神經網路中常用的矩陣運算。

### 🛠️ 步驟1：從原始碼編譯安裝KleidiAI

```bash
# 下載KleidiAI原始碼
$ git clone https://git.gitlab.arm.com/kleidi/kleidiai.git && cd kleidiai

# 配置編譯環境（針對ARM64架構）
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake -S . -B build/

# 開始編譯
$ cmake --build ./build && cd ~
```

💡 **解說**：這個編譯過程會針對您的ARM處理器進行優化，提升數學運算效能。

---

## 🎯 下一步：開始AI模型部署

恭喜！您已經完成了Genio開發環境的基礎設定。現在您可以：

1. **轉換模型格式**：將您的TensorFlow/PyTorch模型轉換為TensorFlow Lite格式
2. **效能測試**：使用專案中的benchmark工具測試不同硬體加速選項
3. **部署應用**：開發您的邊緣AI應用程式

## 🔗 相關資源

- [模型轉換教學](https://github.com/R300-AI/ITRI-AI-Hub/tree/main/Model-Zoo)
- [效能基準測試指南](../demo/ultralytics_tutorial.md)
- [常見問題排解](../demo/async_streaming_tutorial.md)

## 💡 給AI學生的實用提醒

**從雲端到邊緣的關鍵差異**：
- ⏱️ **延遲優先**：邊緣AI注重推論速度，而非訓練吞吐量
- 💾 **資源受限**：記憶體和運算能力有限，需要模型量化和優化
- 🔋 **功耗考量**：電池供電設備需要考慮能源效率
- 📱 **即時性**：許多應用需要毫秒級的回應時間

**成功的關鍵**：
1. 先在雲端/PC完成模型訓練和驗證
2. 使用TensorFlow Lite進行模型轉換和量化
3. 在Genio上進行效能測試和優化
4. 根據硬體特性調整模型架構
# Genio EVK 開發環境設定指南

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
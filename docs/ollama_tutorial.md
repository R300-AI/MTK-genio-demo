## Deploy Language Process Applications Using Ollama

### Installation
第一步、下載並安裝Ollama框架
  ```bash
  $ curl -fsSL https://ollama.com/install.sh | sh
  ```

第二步、從[Models](https://ollama.com/search)找到合適的模型及尺寸, 並透過`Ollama pull`命令下載pre-built模型
  ```bash
  $ ollama run llama3.2:1b
  ```

第三步、安裝Python SDK
  ```bash
  $ pip install ollama
  ```

### Usage
#### Command-line

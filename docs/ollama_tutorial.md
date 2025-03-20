# Deploy Language Process Applications Using Ollama

## Installation
第一步、下載並安裝Ollama框架
  ```bash
  $ curl -fsSL https://ollama.com/install.sh | sh
  ```

第二步、從[Models](https://ollama.com/search)找到合適的模型及尺寸, 並透過`Ollama pull`命令下載pre-built模型
  ```bash
  $ ollama pull llama3.2:1b
  ```

第三步、安裝Python library
  ```bash
  $ pip install ollama
  ```

## Usage
### Command-Line Response
  透過`ollama run`激活問答環境，並輸入你想提問的內容
  ```bash
  $ ollama run llama3.2:1b
  ```

### Python Streaming Responses
  透過`vim`文字編輯器建立`streaming.py`檔，並執行以下內容
  ```python
  from ollama import chat
  
  stream = chat(
      model='llama3.2',
      messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
      stream=True,
  )
  
  for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
 ```

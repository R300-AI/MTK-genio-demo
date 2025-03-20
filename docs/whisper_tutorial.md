# Installation
  ```bash
  $ pip install -U openai-whisper
  $ sudo apt update && sudo apt install ffmpeg
  $ pip install setuptools-rust
  ```

# Python Usage
從[這裡](https://github.com/openai/whisper)找尋模型，並且執行
```python
import whisper, time

model_name = "tiny"
model = whisper.load_model(model_name)

t = time.time()
result = model.transcribe("./data/time.mp3")
print(result["text"], f", Real-Time Factor: {(time.time() - t)/3}")
```

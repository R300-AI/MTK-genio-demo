# Deploy Whisper Applications Using OpenAI Library

## Installation
  To use the Whisper library, you need to install it along with some dependencies. Follow these steps:

  ```bash
  $ pip install -U openai-whisper
  $ sudo apt update && sudo apt install ffmpeg
  $ pip install setuptools-rust
  ```

## Python Usage
From the [Whisper GitHub repository](https://github.com/openai/whisper), select the desired model and test it using the example audio files provided in the repository.
```python
import whisper, time

model_name = "tiny"
model = whisper.load_model(model_name)

t = time.time()
result = model.transcribe("./data/time.mp3")
print(result["text"], f", Real-Time Factor: {(time.time() - t)/3}")
```

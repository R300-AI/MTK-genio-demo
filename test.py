import cv2, time
from ultralytics import YOLO
import queue
from threading import Thread

def thread_safe_predict(frame, buffer):
    print('predict')
    results = model.predict(frame, stream=True, verbose=False)
    print('put')
    buffer.put(results)

video_path = "/content/reverse.mp4"
cap = cv2.VideoCapture(video_path)
buffer = queue.Queue()
i = 0
t = time.time()
while cap.isOpened() and i <= 32:
    _, frame = cap.read()
    Thread(target=thread_safe_predict, args=(frame, buffer)).start()
    if not buffer.empty():
      results = buffer.get()
      #results[0].show()
      i += 1
print(t - time.time())

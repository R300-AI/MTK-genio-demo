from ultralytics import YOLO
import queue, threading, time, psutil, cv2

class YOLOThreadSafe(threading.Thread):
    def __init__(self, identifier=0):
        super().__init__()
        self.occupied = False
        self.running = True
        self.identifier = identifier
        self.model = YOLO("./models/yolov8n_float32.tflite", task='detect')
        self.model.predict('./data/bus.jpg', verbose=False)    # Used to initialize the interpreter
        
    def run(self):
        while self.running:
            if inputs.qsize() > 0 and not self.occupied:
                self.occupied = True
                data = inputs.get()
                order, frame = data[0], data[1]

                results = self.model.predict(frame, verbose=False)
                
                result = results[0].plot()
                result = cv2.putText(result, f'[Processor {self.identifier}] CPU usage: {psutil.cpu_percent(1)}, RAM memory %:{psutil.virtual_memory().percent})', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                outputs[order] = cv2.resize(result, (600, 600))
                del data
                self.occupied = False
        
inputs = queue.Queue()
outputs = {}
THREADS = {}
MAX_THREAD = 30
for i in range(MAX_THREAD):
    THREADS[i] = YOLOThreadSafe(i)
    THREADS[i].start()

cap = cv2.VideoCapture("./data/serve.mp4")
serve, order = 0, 0
while True:
    # INPUT
    occupied = [THREADS[i].occupied for i in range(MAX_THREAD)]
    if False in occupied:
        ret, frame = cap.read()
        if ret == True:
            inputs.put([order, frame])
            order += 1
            
    # OUTPUT
    if serve in outputs:
        result = outputs[serve]
        cv2.imshow('streaming', result)
        del outputs[serve]
        serve += 1
        if cv2.waitKey(1) == ord('q'):
            break 
    if serve >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break
                
for i in range(MAX_THREAD):
    THREADS[i].running = False
cap.release() 
cv2.destroyAllWindows()   

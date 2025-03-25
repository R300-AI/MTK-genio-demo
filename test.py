import queue, threading, time

def worker(i):
    for i in range(10):
        time.sleep(1)
    return f'{i} finished'

class YOLOThreadSafe(threading.Thread):
    def __init__(self, identifier=0):
        super().__init__()
        self.identifier = identifier
        self.running = False
        self.engine = worker
        
    def run(self):
        while True:
            if inputs.qsize() > 0 and not self.running:
                print(f'Processor {self.identifier}, Get a new task')
                self.running = True
                ret = self.engine(self.identifier)
                outputs.put(ret)
                print(f'Processor {self.identifier}, Finished a task')
                self.running = False

inputs = queue.Queue()
outputs = queue.Queue()

workers = {}
max_workers = 5
for i in range(max_workers):
    workers[i] = YOLOThreadSafe(i)
    workers[i].start()

for i in range(100):
    inputs.put(i)
    if outputs.qsize() > 0:
        print('main output:', outputs.get())
    time.sleep(1)
    
import cv2
import logging

# 使用與 pipeline 相同的 logger
logger = logging.getLogger('gstreamer_demo')

class Consumer:
    def __init__(self, window_name="", monitor=None, display_size=None):
        logger.info(f"[CONSUMER] window name - {window_name}")
        self.window_name = window_name
        self.monitor = monitor
        self.display_size = display_size
        self.frame_count = 0

    def consume(self, frame):
        self.frame_count += 1
        
        try:
            if self.display_size is not None:
                frame = cv2.resize(frame, self.display_size)
            if self.monitor:
                frame = self.monitor.draw_info(frame)

            cv2.imshow(self.window_name, frame)
            
            if self.monitor:
                self.monitor.count_consumed()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise StopIteration
                
        except Exception as e:
            logger.error(f"CONSUMER_CONSUME: Error displaying frame {self.frame_count}: {e}")

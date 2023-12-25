import time 
import cv2 
import threading

class Camera():
    def __init__(self):
        self.FRAME_HEIGHT = 640
        self.FRAME_WIDTH = 480
        self.FRAME_RATE = 10

        self.thread_lock = threading.Lock()

        self.current_frame = None
        self.has_frame = True

        self.is_running = False 
        self.value = 0
        self.start()

    def updateFrame(self):
        camera = cv2.VideoCapture(0)
        time.sleep(0.1)

        while self.is_running:
            self.has_frame, self.current_frame = camera.read() 
        camera.release()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.updateFrame)
        self.thread.start()
    def stop(self):
        self.is_running = False
    def read(self):
        return self.has_frame, self.current_frame
    def getValue(self):
        return self.value


if __name__ == "__main__":
    camera = Camera()
    time.sleep(0.1)
    while True:
        try:
            with camera.thread_lock:
                ret, frame = camera.read()
                if not ret:
                    continue
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1) & 0xFF
                time.sleep(1)

                if key == ord('q'):
                    camera.stop()
                    break
        except Exception as e:
            print(e)
            pass
    cv2.destroyAllWindows()
    

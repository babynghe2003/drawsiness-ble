from picamera.array import PiRGBArray 
from picamera import PiCamera 
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
        self.is_running = False 
        self.value = 0
        self.start()

    def updateFrame(self):
        camera = PiCamera()
        camera.resolution = (self.FRAME_HEIGHT, self.FRAME_WIDTH)
        camera.framerate = self.FRAME_RATE 
        raw_capture = PiRGBArray(camera, size=(self.FRAME_HEIGHT, self.FRAME_WIDTH))
        time.sleep(0.1)

        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            if not self.is_running:
                break
            image = frame.array
            image = cv2.flip(image,1)
            self.current_frame = image.copy()
            raw_capture.truncate(0)
        camera.close()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.updateFrame)
        self.thread.start()
    def stop(self):
        self.is_running = False
    def getFrame(self):
        return self.current_frame
    def getValue(self):
        return self.value


if __name__ == "__main__":
    camera = Camera()
    time.sleep(0.1)
    while True:
        try:
            with camera.thread_lock:
                frame = camera.getFrame()
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1) & 0xFF
                time.sleep(2)

                if key == ord('q'):
                    camera.stop()
                    break
        except Exception as e:
            print(e)
            pass
    cv2.destroyAllWindows()
    

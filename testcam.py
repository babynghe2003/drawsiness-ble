from picamera.array import PiRGBArray 
from picamera import PiCamera 
import time 
import cv2 

#Instantiate and configure picamera 
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 40
raw_capture = PiRGBArray(camera, size=(640, 480))
#let camera module warm up 
time.sleep(0.1)


start = time.time()
try:
    #Capture image array objects as video frames
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        image = cv2.flip(image,1)
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        raw_capture.truncate(0)
        #Save frames to output video
        #if 'q' is pressed, close OpenCV window and end video recording
        if key == ord('q'):
            break
except Exception as e:
    print(e)
    print(time.time() - start)
finally:
    camera.close()
    cv2.destroyAllWindows()

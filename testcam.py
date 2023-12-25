# from picamera.array import PiRGBArray 
# from picamera import PiCamera 
import time 
import cv2 

vid_cap = cv2.VideoCapture(0) 

while True:
    ret, frame = vid_cap.read()
    cv2.imshow("Frame Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vid_cap.release()
cv2.destroAllWindows()

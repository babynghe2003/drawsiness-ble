import cv2 as cv
import mediapipe as mp
import math
import threading
import utils
import numpy as np
import time
from camera_capture import Camera
import RPi.GPIO as GPIO

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]



class DrowsinessDetector():
    def __init__(self,show = False, buzz = False) -> None:
        self.BLINK_EYES_FRAME = 3  # if drowsy counter frame > 5, then dblink
        self.DROWSY_BLINKS = 5 # if blink frame > 5, then drowsy 
        self.SLEEPING_BLINKS = 12 # if blink frame > 10, then sleeping

        self.BLINK_RATIO = 3.8

        self.frame_counter = 0
        self.show = show
        self.buzz = buzz

        self.awake_counter = 0
        self.drowsy_counter = 0

        self.total_blinks = 0
        self.last_blink_frame = 0

        self.is_running = True
        self.thread_lock = threading.Lock()

        self.is_alert = False 
        self.is_drowsy = False
        self.is_sleeping = False

        self.showimg = False

        # self.cap = cv.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh

        self.FONTS = cv.FONT_HERSHEY_PLAIN

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        self.buzzer = 23
        GPIO.setup(self.buzzer, GPIO.OUT)

        self.isBuzzing = False

    def _landmarksDetection(self, img, results, draw=False):
        img_h, img_w, img_c = img.shape
        mesh_coord = [(int(point.x * img_w), int(point.y * img_h)) for point in results.multi_face_landmarks[0].landmark]
        if draw:
            for idx, coord in enumerate(mesh_coord):
                cv.circle(img, coord, 1, (0, 255, 0), -1)
                cv.putText(img, str(idx), coord, cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
        return mesh_coord
    
    def _euclaideanDistance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def _blinkRatio(self, img, landmarks, right_indices, left_indices):
        rh_right = landmarks[right_indices[0]] 
        rh_left = landmarks[right_indices[8]]
        rh_top = landmarks[right_indices[12]]
        rh_bottom = landmarks[right_indices[4]]

        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]

        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rh_distance = self._euclaideanDistance(rh_right, rh_left)
        rv_distance = self._euclaideanDistance(rh_top, rh_bottom)

        lv_distance = self._euclaideanDistance(lv_top, lv_bottom)
        lh_distance = self._euclaideanDistance(lh_right, lh_left)

        reratio = rh_distance / rv_distance
        leratio = lh_distance / lv_distance

        return reratio, leratio
    
    def drawsinessDetect(self):
        try:
            self.camera = Camera()
            with self.mp_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
                while self.is_running:
                    try:
                        success, frame = self.camera.read()
                        if not success:
                            self.is_running = False
                            break

                        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
                        frame_height, frame_width = frame.shape[:2]
                       
                        self.frame_counter += 1
                        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                        result = face_mesh.process(rgb_frame)
                        if result.multi_face_landmarks:
                            mesh_coords = self._landmarksDetection(frame, result, draw=True)
                            ratio_left, ratio_right = self._blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

                            if (ratio_left + ratio_right) / 2 > self.BLINK_RATIO: 
                                self.drowsy_counter += 1
                                self.awake_counter = 0
                                print(".")
                                if self.show:
                                    utils.colorBackgroundText(frame, f'BLINK', self.FONTS, 2.7, (300,100), 2, utils.YELLOW, pad_x = 6, pad_y = 6,)  
                            else:
                                self.awake_counter += 1
                                self.drowsy_counter = 0
                                if self.is_alert:
                                    self.is_alert = False
                                if self.is_drowsy:
                                    self.is_drowsy = False
                                if self.is_sleeping:
                                    self.is_sleeping = False

                            if self.drowsy_counter > self.BLINK_EYES_FRAME:
                                print("Blink")
                                self.drowsy_counter = 0
                                self.total_blinks += 1
                                self.last_blink_frame = self.frame_counter

                            if self.frame_counter - self.last_blink_frame > 10:
                                self.total_blinks = 0 
                                self.last_blink_frame = self.frame_counter
                                if self.total_blinks < 0:
                                    self.total_blinks = 0
                                if self.isBuzzing:
                                    GPIO.output(self.buzzer, GPIO.LOW)

                            if self.total_blinks > self.DROWSY_BLINKS:
                                print("Drowsy")
                                if self.buzz:
                                    GPIO.output(self.buzzer, GPIO.HIGH)
                                    self.isBuzzing = True
                                self.is_alert = True
                                self.is_drowsy = True

                            if self.total_blinks > self.SLEEPING_BLINKS:
                                print("Sleeping")
                                self.is_sleeping = True
                                self.total_blinks = self.SLEEPING_BLINKS + 1

                        if self.show:
                            utils.colorBackgroundText(frame,  f'Total Blinks: {self.total_blinks}', self.FONTS, 1.7, (30,150),2)
                            utils.colorBackgroundText(frame, f'Ratio: {round(ratio_left,2)} {round(ratio_right,2)}', self.FONTS, 1.7, (30,100), 2, utils.PINK, utils.YELLOW)
                            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                            cv.imshow('frame', frame)
                            key = cv.waitKey(2)
                            if key==ord('q') or key ==ord('Q'):
                                self.stop()
                                break
                    except Exception as e:
                        print(e)
                        pass
                
        except Exception as e:
            pass
        finally:
            cv.destroyAllWindows()
            self.camera.stop()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.drawsinessDetect)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()

    def isAlert(self):
        return self.is_alert

    def isDrowsy(self):
        return self.is_drowsy

    def isSleeping(self):
        return self.is_sleeping


if __name__ == "__main__":
    drowsiness_detector = DrowsinessDetector(buzz = True)
    drowsiness_detector.start()


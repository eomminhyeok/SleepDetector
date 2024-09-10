import cv2
import dlib
from functools import wraps
from scipy.spatial import distance
import time
from gpiozero import RGBLED
from time import sleep
from RPi_I2C_LCD_driver import RPi_I2C_driver
import threading

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)
cap.set(3, 256)
cap.set(4, 144)

lcd = RPi_I2C_driver.lcd(0x27)
led = RGBLED(red=16, green=20, blue=21)

# dlib 인식 모델 정의
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# gpio 셋팅
lastsave = 0
output_pin = 18

def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        time.sleep(0.05)
        global lastsave
        if time.time() - lastsave > 5:
            lastsave = time.time()

        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

@counter
def close():
    cv2.putText(frame, "DROWSY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

def usual():
    cv2.putText(frame, "Good", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def eye_close():
    cv2.putText(frame, "Warning", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 255, 255), 2)

def deep_sleep():
    cv2.putText(frame, "Dangerous", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):  # 오른쪽 눈 감지
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):  # 왼쪽 눈 감지
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)

        if EAR < 0.3:
            close.count += 0.1
            print(f'close count : {close.count}')

            if close.count >= 4:
                if led.red != 1:
                    deep_sleep()
                    print("Dangerous")
                    led.blue = 0
                    led.green = 0
                    led.red = 1
                    lcd.clear()
                    lcd.setCursor(1, 0)
                    lcd.print("!!!!DANGER!!!!")

            elif close.count >= 2:
                if led.blue != 1:
                    eye_close()
                    print("Driver is sleeping")
                    led.green = 0
                    led.red = 0
                    led.blue = 1
                    lcd.clear()
                    lcd.setCursor(1, 0)
                    lcd.print("  !!Warning!!   ")

        else:
            if led.green != 1:
                usual()
                print("Perfect")
                led.blue = 0
                led.red = 0
                led.green = 1
                lcd.clear()
                lcd.print("Good!")
            close.count = 0

        print(EAR)

    cv2.imshow("Sleeping detection", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import time

cap = cv2.VideoCapture(1)
def capture_webcam ():
    global cap

    #ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    #ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240).

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    if cap.isOpened():
        ret, rgb_frame = cap.read()
        if ret is False:
            rgb_frame = np.zeros((width, height, 3), np.uint8)
    else:
        width = 224
        height = 224
        rgb_frame = np.zeros((width, height, 3), np.uint8)

    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)



    return rgb_frame, gray_frame

label = '33'
path = './images/'
i = 0
cnt = 0

for i in range(1, 100):
    data = capture_webcam()
    cnt = str(i)
    rgb = os.path.join(path + '/rgb/' + label)
    gray = os.path.join(path + '/gray/' + label)
    cv2.imwrite(os.path.join(rgb, 'IMG' + cnt + '.jpg'), data[0])
    cv2.imwrite(os.path.join(gray, 'IMG' + cnt + '.jpg'), data[1])
    time.sleep(0.15)
    #cv2.imshow('test', data[0])
    #cv2.imshow('test2', data[1])

print('______________________________________')
time.sleep(5)

i = 100

for i in range(100, 200):
    data = capture_webcam()
    cnt = str(i)
    rgb = os.path.join(path + '/rgb/' + label)
    gray = os.path.join(path + '/gray/' + label)
    cv2.imwrite(os.path.join(rgb, 'IMG' + cnt + '.jpg'), data[0])
    cv2.imwrite(os.path.join(gray, 'IMG' + cnt + '.jpg'), data[1])
    time.sleep(0.15)
    #cv2.imshow('test', data[0])
    #cv2.imshow('test2', data[1])
print('finished')
cap.release()
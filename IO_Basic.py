import numpy as np
import cv2
from pathlib import Path
import os

import random
import math
import multiprocessing as mp

def open_picture (pfad_und_dateiendung =""):

    rgb = cv2.imread(pfad_und_dateiendung)
    if rgb is None:
        rgb = np.zeros((100, 100, 3), np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    return rgb, gray

def capture_webcam ():
    cap = cv2.VideoCapture(0)

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

    cap.release()

    return rgb_frame, gray_frame

def get_video_frame (pfad_und_dateiendung ="", framenumber = 0):
    cap = cv2.VideoCapture(pfad_und_dateiendung)

    # ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    # ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240).

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenumber)
        ret, rgb_frame = cap.read()
        if ret is False:
            rgb_frame = np.zeros((width, height, 3), np.uint8)
    else:
        rgb_frame = np.zeros((width, height, 3), np.uint8)

    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    return rgb_frame, gray_frame

def get_frame_numbers (pfad_und_dateiendung = ""):
    frame_nummber = 0
    cap = cv2.VideoCapture(pfad_und_dateiendung)
    if cap.isOpened():
        while True:
            ret, rgb_frame = cap.read()
            if ret is False:
                break
            frame_nummber += 1
        cap.release()
    return frame_nummber
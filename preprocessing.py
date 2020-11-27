import numpy as np
import cv2
from pathlib import Path
import os

import IO_Basic as IO

import random
import math
import multiprocessing as mp

def bild_mit_simulationsfehler (image = np.zeros((100, 100), np.uint8), simulationsfehler = 0):
    (width, height) = image.shape #dimension
    pixel = width * height

    try:
        if pixel <= 0:
            print("Bildpunkte in 'bild_mit_simulationsfehler' hat falschen Wert: ", pixel,"\n")
            return None
        if width <= 0:
            print("Breite in 'bild_mit_simulationsfehler' hat falschen Wert: ", width,"\n")
            return None
        if height <= 0 :
            print("Hoehe in 'bild_mit_simulationsfehler' hat falschen Wert: ", height,"\n")
            return None
        # if dimension >1:
        #     print("Dimension in 'bild_mit_simulationsfehler' hat falschen Wert: ", dimension,"\n")
        #     return None

        badpixel = pixel * simulationsfehler / 100

        if simulationsfehler > 0:
            for i in range(badpixel):
                x = random.randint(0, width)
                y = random.randint(0, height)
                #pixel = x + width * y
                if image[x][y] < 127:
                    image[x][y] = 255
                else:
                    image[x][y] = 0
        elif simulationsfehler < 0:
            random = abs(simulationsfehler)
            for x in range(width):
                for y in range(height):
                    error = gauss_error(random)
                    grauwert = error
                    grauorg = image[x][y]
                    if simulationsfehler > -255:
                        imagegrey = grauorg + grauwert
                    else:
                        imagegrey = grauwert
                    if imagegrey < 0:
                        image[x][y] = 0
                    elif imagegrey > 255:
                        image[x][y] = 255
                    else:
                        image[x][y] = imagegrey

        return image
    except:
        print("Fehler in bild_mit_simulationsfehler!\n")
        return None

def gauss_error(sigma = 0):
    varianz = sigma

    try:
        rx = random.random()
        ry = random.random()
        if rx == 0.0:
            rx = 1.0 / 32767
        rx1= math.sqrt(-2 * math.log(rx)) * math.cos(2 * math.pi * ry) * varianz
    except:
        print("Fehker in gauss_error!\n")
        return -1.0;

    return rx1


path_picture = os.path.abspath("Bilder\\20121210_152659.jpg")
#path_video = os.path.abspath("Bilder\\20121210_152721.mp4")
rgb, gray = IO.open_picture(path_picture)

gaus_blur=cv2.GaussianBlur(gray,(3,3),0)
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
cv2.imshow('sobelx-Bild', sobelx)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
cv2.imshow('sobely-Bild', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Adaptive Threshold
adaptiv_thresh = cv2.adaptiveThreshold(gaus_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
cv2.imshow('adaptiv_thresh-Bild', adaptiv_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Binarisieren mit der Otsu-Methode
ret, otsu= cv2.threshold(gaus_blur,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('otsu_bin-Bild', otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()


edges = cv2.Canny(gaus_blur,ret,210)
cv2.imshow('Test-Bild', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Test von simplen sachen
#test = bild_mit_simulationsfehler(gray, -10)
# cv2.imshow('Test-Bild', test)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#rgb, gray = IO.capture_webcam ()

#x = IO.get_frame_numbers(path_video)
#print(x)

#rgb1, gray1 = IO.get_video_frame(path_video, 1)
#rgb2, gray2 = IO.get_video_frame(path_video, 84)

#test = gray1.copy()

#cv2.addWeighted(gray1, 0.5, gray2, 0.5, 0, test)

#rgb_resize = cv2.resize(test, (960, 540))
#cv2.imshow('RGB-Bild', rgb_resize)

#for frame in range(x):
#    rgb, gray = IO.get_video_frame(path_video, frame)
#    print(frame)

#    rgb_resize = cv2.resize(rgb, (960, 540))
#    gray_resize = cv2.resize(gray, (960, 540))
#    cv2.imshow('RGB-Bild', rgb_resize)
#    cv2.imshow('Gray-Bild', gray_resize)
#    cv2.waitKey(0)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

rgb, gray = IO.open_picture(path_picture)

#
blur = cv2.GaussianBlur(gray, (5, 5), 10)

cv2.imshow('Binaer-Bild', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

thresh, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

binary_show = binary#cv2.resize(binary, (960, 540))
cv2.imshow('Binaer-Bild', binary_show)

cv2.waitKey(0)
cv2.destroyAllWindows()
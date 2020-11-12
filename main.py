import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Open the image
    img = cv2.imread('frame231.png')
    print(img.size)
    # img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    cv2.imshow('Original', img)

    # Create the gray-scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create the L-Alpha-Beta image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow('Lab', lab)

    # Mask the skin color
    skin_color_mask = cv2.inRange(lab, (0, 134, 125), (255, 182, 182))
    gray_masked = cv2.bitwise_and(gray, gray, mask=skin_color_mask)
    # cv2.imshow('LAB masked default', lab_masked)

    # Apply opening filter
    OPENING_K_SIZE = 9
    gray_opened = cv2.morphologyEx(gray_masked, cv2.MORPH_OPEN, np.ones((OPENING_K_SIZE, OPENING_K_SIZE), np.uint8))
    cv2.imshow('After opening filter', gray_opened)

    plt.hist(gray_opened.ravel(), 256, [1, 256])
    plt.show()

    ret, binary = cv2.threshold(gray_opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Otsu', binary)

    # Edge detection
    canny_lab = cv2.Canny(gray_opened, 30, 123)
    cv2.imshow('Canny Opening [47,123]', canny_lab)

    # Apply closing filter
    # CLOSING_K_SIZE = 9
    # lab_closed = cv2.morphologyEx(lab_masked, cv2.MORPH_CLOSE, np.ones((CLOSING_K_SIZE, CLOSING_K_SIZE), np.uint8))
    # cv2.imshow('After closing filter', lab_closed)

    # cv2.namedWindow('LAB')
    # cv2.createTrackbar('l_min', 'LAB', 0, 255, lambda x: x)
    # cv2.createTrackbar('a_min', 'LAB', 0, 255, lambda x: x)
    # cv2.createTrackbar('b_min', 'LAB', 0, 255, lambda x: x)
    # cv2.createTrackbar('l_max', 'LAB', 255, 255, lambda x: x)
    # cv2.createTrackbar('a_max', 'LAB', 255, 255, lambda x: x)
    # cv2.createTrackbar('b_max', 'LAB', 255, 255, lambda x: x)
    #
    # while True:
    #     l_min = cv2.getTrackbarPos('l_min', 'LAB')
    #     a_min = cv2.getTrackbarPos('a_min', 'LAB')
    #     b_min = cv2.getTrackbarPos('b_min', 'LAB')
    #     l_max = cv2.getTrackbarPos('l_max', 'LAB')
    #     a_max = cv2.getTrackbarPos('a_max', 'LAB')
    #     b_max = cv2.getTrackbarPos('b_max', 'LAB')
    #     skin_color_mask = cv2.inRange(lab, (l_min, a_min, b_min), (l_max, a_max, b_max))
    #     lab_masked = cv2.bitwise_and(lab, lab, skin_color_mask=skin_color_mask)
    #     cv2.imshow('LAB masked', lab_masked)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break

    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('Block Size', 'Trackbar', 23, 100, lambda x: x)
    cv2.createTrackbar('Constant', 'Trackbar', 4, 100, lambda x: x)

    # Edge detection
    canny = cv2.Canny(gray, 47, 123)
    cv2.imshow('Canny [47,123]', canny)
    hand_roi = canny

    while True:
        block_size = cv2.getTrackbarPos('Block Size', 'Trackbar')
        block_size = max(block_size if block_size % 2 == 1 else block_size - 1, 3)
        c = cv2.getTrackbarPos('Constant', 'Trackbar')
        th3 = cv2.adaptiveThreshold(gray_opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
        cv2.imshow('Adaptive Thresholding', th3)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Prepare region of interest of hand for neuronal network
    hand_roi_resized = cv2.resize(hand_roi, (224, 224))
    blob = cv2.dnn.blobFromImage(hand_roi_resized, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123), swapRB=True)
    print("First Blob: {}".format(blob.shape))

    # Wait for the user to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

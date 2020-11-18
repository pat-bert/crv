from glob import glob

import cv2
import numpy as np

from check_label import ROOT_PATH

LMIN, LMAX = 0, 255
AMIN, AMAX = 128, 182
BMIN, BMAX = 132, 182

if __name__ == '__main__':
    # Get a random image from the set of allowed labels
    image_files = glob(str(ROOT_PATH) + '/images_320-240_1/*/*/Color/*/*.jp*g')
    img_path = np.random.choice(image_files)
    img_original = cv2.imread(img_path)

    # Enhance the contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Transform to other color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    row1 = np.hstack((img_original, img))
    row2 = np.hstack((lab, gray_bgr))
    res = np.vstack((row1, row2))
    cv2.imshow(f'Original | Original CLAHE | LAB CLAHE | Gray CLAHE', res)

    # Mask the skin color
    skin_color_mask = cv2.inRange(lab, (LMIN, AMIN, BMIN), (LMAX, AMAX, BMAX))
    gray_masked = cv2.bitwise_and(gray, gray, mask=skin_color_mask)

    cv2.namedWindow('LAB', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('l_min', 'LAB', LMIN, 255, lambda x: x)
    cv2.createTrackbar('a_min', 'LAB', AMIN, 255, lambda x: x)
    cv2.createTrackbar('b_min', 'LAB', BMIN, 255, lambda x: x)
    cv2.createTrackbar('l_max', 'LAB', LMAX, 255, lambda x: x)
    cv2.createTrackbar('a_max', 'LAB', AMAX, 255, lambda x: x)
    cv2.createTrackbar('b_max', 'LAB', BMAX, 255, lambda x: x)

    while True:
        l_min = cv2.getTrackbarPos('l_min', 'LAB')
        a_min = cv2.getTrackbarPos('a_min', 'LAB')
        b_min = cv2.getTrackbarPos('b_min', 'LAB')
        l_max = cv2.getTrackbarPos('l_max', 'LAB')
        a_max = cv2.getTrackbarPos('a_max', 'LAB')
        b_max = cv2.getTrackbarPos('b_max', 'LAB')
        skin_color_mask = cv2.inRange(lab, (l_min, a_min, b_min), (l_max, a_max, b_max))
        lab_masked = cv2.bitwise_and(lab, lab, mask=skin_color_mask)
        cv2.imshow('LAB masked', lab_masked)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Prepare region of interest of hand for neuronal network

    # Wait for the user to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

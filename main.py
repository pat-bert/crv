from glob import glob

import cv2
import numpy as np

from check_label import ROOT_PATH

LMIN, LMAX = 0, 255
AMIN, AMAX = 131, 165
BMIN, BMAX = 132, 178
BLUR_SIZE = 3
EROSION_SIZE = 9
closing_size = 3


def make_uneven(val):
    val = max(3, val)
    return val if val % 2 == 1 else val + 1


if __name__ == '__main__':
    # Get a random image from the set of allowed labels
    image_files = glob(str(ROOT_PATH) + '/images_320-240_1/*/*/Color/*/*.jp*g')

    while True:
        img_path = np.random.choice(image_files)
        img_original = cv2.imread(img_path)
        print(f'Using image: {img_path}')

        # Enhance the contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        # Transform to other color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        row1 = np.hstack((img_original, img, lab, gray_bgr))

        # Create trackbars
        cv2.namedWindow('Trackbars', cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('a_min', 'Trackbars', AMIN, 255, lambda x: x)
        cv2.createTrackbar('b_min', 'Trackbars', BMIN, 255, lambda x: x)
        cv2.createTrackbar('a_max', 'Trackbars', AMAX, 255, lambda x: x)
        cv2.createTrackbar('b_max', 'Trackbars', BMAX, 255, lambda x: x)
        cv2.createTrackbar('erosion', 'Trackbars', EROSION_SIZE, 30, lambda x: x)
        cv2.createTrackbar('blur', 'Trackbars', BLUR_SIZE, 10, lambda x: x)

        while True:
            # Get trackbar values
            a_min = cv2.getTrackbarPos('a_min', 'Trackbars')
            b_min = cv2.getTrackbarPos('b_min', 'Trackbars')
            a_max = cv2.getTrackbarPos('a_max', 'Trackbars')
            b_max = cv2.getTrackbarPos('b_max', 'Trackbars')
            erosion_size = make_uneven(cv2.getTrackbarPos('erosion', 'Trackbars'))
            blur_size = make_uneven(cv2.getTrackbarPos('blur', 'Trackbars'))

            # Mask the skin color
            skin_color_mask = cv2.inRange(lab, (LMIN, a_min, b_min), (LMAX, a_max, b_max))
            gray_masked = cv2.bitwise_and(gray, gray, mask=skin_color_mask)

            # Binarize the image
            ret, gray_binary = cv2.threshold(gray_masked, 1, 255, cv2.THRESH_BINARY)
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            gray_binary = cv2.morphologyEx(gray_binary, cv2.MORPH_OPEN, kernel)
            gray_binary = cv2.morphologyEx(gray_binary, cv2.MORPH_CLOSE, kernel)

            # Blur the image before edge detection
            gray_blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), sigmaX=2)
            v = np.median(gray)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))

            # Detect edges in the grayscale image
            gray_edges = cv2.Canny(gray_blurred, lower, upper, apertureSize=3)
            kernel = np.ones((closing_size, closing_size), np.uint8)
            gray_edges = cv2.morphologyEx(gray_edges, cv2.MORPH_DILATE, kernel)

            # Subtract the edges from the binary image
            gray_edges_inv = cv2.bitwise_not(gray_edges)
            gray_binary = cv2.bitwise_and(gray_binary, gray_edges_inv)

            # Generate a copy of the original within the loop
            img_box = np.copy(img_original)

            # Find contours in gray image
            box_cnt = 0
            contours, hierarchy = cv2.findContours(gray_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(gray_binary.shape)
            TOTAL_AREA = np.prod(gray_binary.shape[0:2])
            MAX_AREA = 0.7 * TOTAL_AREA
            MIN_AREA = 0.03 * TOTAL_AREA
            print(f'Area of boxes must be within ({MIN_AREA},{MAX_AREA})')
            print(f'Found {len(contours)} contours')
            if len(contours) > 1:
                min_area_act = min(cv2.contourArea(cnt) for cnt in contours)
                max_area_act = max(cv2.contourArea(cnt) for cnt in contours)
                print(f'Minimum area found {min_area_act}, maximum area found {max_area_act}')

            # Draw contours from binary image
            cv2.drawContours(img_box, contours, -1, (255, 0, 0), 3)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Draw bounding box if area of contour is sufficient
                if MAX_AREA > area > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                    box_cnt += 1
                    print(f'Found {box_cnt} boxes so far')

            row2 = np.hstack((
                cv2.cvtColor(gray_masked, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(gray_binary, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2BGR),
                img_box
            ))
            res = np.vstack((row1, row2))
            cv2.imshow('<Original | Original CLAHE | LAB CLAHE | Gray CLAHE> <Skin Masked | Binary | Edges | Boxes>',
                       res)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        # Prepare region of interest of hand for neuronal network

        cv2.destroyAllWindows()

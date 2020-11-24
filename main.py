import random
from glob import glob

import cv2
import numpy as np

from check_label import ROOT_PATH

# Skin Color Mask
LMIN, LMAX = 0, 255
AMIN, AMAX = 131, 165
BMIN, BMAX = 132, 190

# Filter Kernel Sizes
CANNY_BLUR_SIZE = 5
BINARY_OPEN_CLOSE_SIZE = 9
EDGE_DILATION_SIZE = 3
MIN_FILL_RATIO = 0.3

# Valid Area Range
MIN_AREA_REL, MAX_AREA_REL = 0.03, 0.7


def make_uneven(val):
    val = max(3, val)
    return val if val % 2 == 1 else val + 1


def get_area_thresholds(image):
    """
    Calculate the absolute area thresholds with respect to the total image area
    :param image:
    :return:
    """
    total_area = np.prod(image.shape[0:2])
    return MIN_AREA_REL * total_area, MAX_AREA_REL * total_area


def do_nothing():
    pass


if __name__ == '__main__':
    # Get all image file paths in one of the source folders
    image_files = glob(str(ROOT_PATH) + f'/images_320-240_{random.randint(1, 5)}/*/*/Color/*/*.jp*g')

    while True:
        # Get a random image from the set of allowed labels
        img_path = np.random.choice(image_files)
        img_original = cv2.imread(img_path)

        # Create trackbars
        cv2.namedWindow('Trackbars', cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('Min Alpha', 'Trackbars', AMIN, 255, do_nothing)
        cv2.createTrackbar('Min Beta', 'Trackbars', BMIN, 255, do_nothing)
        cv2.createTrackbar('Max Alpha', 'Trackbars', AMAX, 255, do_nothing)
        cv2.createTrackbar('Max Beta', 'Trackbars', BMAX, 255, do_nothing)
        cv2.createTrackbar('Binary Open/Close Size', 'Trackbars', BINARY_OPEN_CLOSE_SIZE, 30, do_nothing)
        cv2.createTrackbar('Canny Blur', 'Trackbars', CANNY_BLUR_SIZE, 10, do_nothing)
        a_min, a_max, b_min, b_max, binary_kernel_size, canny_blur_size = -1, -1, -1, -1, -1, -1

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

        while True:
            # Update previous value
            a_min_prev = a_min
            a_max_prev = a_max
            b_min_prev = b_min
            b_max_prev = b_max
            binary_kernel_size_prev = binary_kernel_size
            blur_prev = canny_blur_size

            # Get trackbar values
            a_min = cv2.getTrackbarPos('Min Alpha', 'Trackbars')
            b_min = cv2.getTrackbarPos('Min Beta', 'Trackbars')
            a_max = cv2.getTrackbarPos('Max Alpha', 'Trackbars')
            b_max = cv2.getTrackbarPos('Max Beta', 'Trackbars')
            binary_kernel_size = make_uneven(cv2.getTrackbarPos('Binary Open/Close Size', 'Trackbars'))
            canny_blur_size = make_uneven(cv2.getTrackbarPos('Canny Blur', 'Trackbars'))

            # Detect a change in the trackbars
            parameters_changed = any(
                [
                    a_min != a_min_prev,
                    a_max != a_max_prev,
                    b_min != b_min_prev,
                    b_max != b_max_prev,
                    binary_kernel_size != binary_kernel_size_prev,
                    canny_blur_size != blur_prev
                ]
            )

            # Mask the skin color
            skin_color_mask = cv2.inRange(lab, (LMIN, a_min, b_min), (LMAX, a_max, b_max))
            gray_masked = cv2.bitwise_and(gray, gray, mask=skin_color_mask)

            # Binarize the image
            ret, gray_binary = cv2.threshold(gray_masked, 1, 255, cv2.THRESH_BINARY)
            kernel = np.ones((BINARY_OPEN_CLOSE_SIZE, BINARY_OPEN_CLOSE_SIZE), np.uint8)
            gray_binary = cv2.morphologyEx(gray_binary, cv2.MORPH_OPEN, kernel)
            gray_binary = cv2.morphologyEx(gray_binary, cv2.MORPH_CLOSE, kernel)

            # Blur the image before edge detection
            gray_blurred = cv2.GaussianBlur(gray, (canny_blur_size, canny_blur_size), sigmaX=2)

            # Detect edges in the grayscale image
            v = np.average(gray)
            sigma = 0.33
            lower_canny = int(max(0, (1.0 - sigma) * v))
            upper_canny = int(min(255, (1.0 + sigma) * v))
            gray_edges = cv2.Canny(gray_blurred, lower_canny, upper_canny, apertureSize=3)
            kernel = np.ones((EDGE_DILATION_SIZE, EDGE_DILATION_SIZE), np.uint8)
            gray_edges = cv2.morphologyEx(gray_edges, cv2.MORPH_DILATE, kernel)

            # Subtract the edges from the binary image
            gray_edges_inv = cv2.bitwise_not(gray_edges)
            gray_binary = cv2.bitwise_and(gray_binary, gray_edges_inv)

            # Calculate the area limits with respect to the image size
            min_area, max_area = get_area_thresholds(img_original)

            # Find contours in gray image
            img_box = np.copy(img_original)
            box_cnt = 0
            contours, hierarchy = cv2.findContours(gray_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_box, contours, -1, (255, 0, 0), thickness=3)

            # Iterate over the contours to find the correct ones
            for i, cnt in enumerate(contours):
                # Calculate the area of the current contour
                area = cv2.contourArea(cnt)
                valid_area = max_area > area > min_area

                # Calculate the fill factor of the contour
                filled_contour = np.zeros(img_box.shape[0:2], dtype=np.uint8)
                cv2.drawContours(filled_contour, contours, i, 255, thickness=cv2.FILLED)
                total_pixels = np.count_nonzero(filled_contour)

                # Get the pixels that are both within the contour and in the binary image
                valid_fill_ratio = False
                if total_pixels > 200:
                    filled_contour = cv2.bitwise_and(filled_contour, gray_binary)
                    filled_pixels = np.count_nonzero(filled_contour)
                    fill_ratio = filled_pixels / total_pixels
                    valid_fill_ratio = fill_ratio > MIN_FILL_RATIO

                # Draw bounding box if area of contour is sufficient
                if valid_area and valid_fill_ratio:
                    box_cnt += 1
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                    cv2.imshow(f'Filled Contour AND Binary Image ({box_cnt})', filled_contour)

            # Debugging information
            if parameters_changed:
                print(80 * '=')
                print('Debug information:')
                print(f'Using image: {img_path}')
                print(f'Found {len(contours)} contours in blurred gray-scale.')
                contours2, _ = cv2.findContours(gray_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                print(f'Found {len(contours)} contours in binary image.')
                print(f'Canny Avg {v:.3f}, Sigma {sigma:.3f}')
                print(f'Canny thresholds: {int(max(0, (1.0 - sigma) * v))};{int(min(255, (1.0 + sigma) * v))}')
                print(80 * '=')

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

import random
from glob import glob

import cv2
import numpy as np
from skimage import color, segmentation, img_as_ubyte
from skimage.future import graph
from skimage.segmentation import mark_boundaries

from check_label import ROOT_PATH

# Skin Color Mask
LMIN, LMAX = 0, 255
AMIN, AMAX = 131, 165
BMIN, BMAX = 132, 214

# Filter Kernel Sizes
EDGE_DILATION_SIZE = 3
MIN_FILL_RATIO = 0.3

# Valid Area Range
MIN_AREA_REL, MAX_AREA_REL = 0.03, 0.7


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])


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


def do_nothing(_):
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
        cv2.createTrackbar('SLIC Thresh', 'Trackbars', 30, 255, do_nothing)

        # Enhance the contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        # Transform to other color spaces
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        while True:
            # Get trackbar values
            a_min = cv2.getTrackbarPos('Min Alpha', 'Trackbars')
            b_min = cv2.getTrackbarPos('Min Beta', 'Trackbars')
            a_max = cv2.getTrackbarPos('Max Alpha', 'Trackbars')
            b_max = cv2.getTrackbarPos('Max Beta', 'Trackbars')
            slic_thresh = cv2.getTrackbarPos('SLIC Thresh', 'Trackbars')

            # Mask the skin color
            skin_color_mask = cv2.inRange(lab, (LMIN, a_min, b_min), (LMAX, a_max, b_max))

            # Segmentation
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            labels = segmentation.slic(img_rgb, n_segments=250, start_label=1, slic_zero=True)
            print(f"SLIC number of segments: {len(np.unique(labels))}")
            out = img_as_ubyte(mark_boundaries(img_rgb.copy(), labels))
            img_segmented = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            # Graph merging
            img_rgb_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=skin_color_mask)
            g = graph.rag_mean_color(img_rgb_masked, labels)
            labels2 = graph.merge_hierarchical(labels, g, thresh=slic_thresh, rag_copy=False, in_place_merge=True,
                                               merge_func=merge_mean_color, weight_func=_weight_mean_color)
            out = color.label2rgb(labels2, img_rgb_masked, kind='avg', bg_label=0)
            out = img_as_ubyte(mark_boundaries(out, labels2, (0, 0, 0)))
            img_merged = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            row1 = np.hstack((img_original, img))
            row2 = np.hstack((img_segmented, img_merged))
            block = np.vstack((row1, row2))
            cv2.imshow('Original and CLAHE enhanced |Segmented and Merged', block)

            # # Binarize the image
            # ret, gray_binary = cv2.threshold(gray_masked, 1, 255, cv2.THRESH_BINARY)
            # kernel = np.ones((BINARY_OPEN_CLOSE_SIZE, BINARY_OPEN_CLOSE_SIZE), np.uint8)
            # gray_binary = cv2.morphologyEx(gray_binary, cv2.MORPH_OPEN, kernel)
            # gray_binary = cv2.morphologyEx(gray_binary, cv2.MORPH_CLOSE, kernel)
            #
            # # Calculate the area limits with respect to the image size
            # min_area, max_area = get_area_thresholds(img_original)

            # # Find contours in gray image
            # img_box = np.copy(img_original)
            # box_cnt = 0
            # contours, hierarchy = cv2.findContours(gray_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img_box, contours, -1, (255, 0, 0), thickness=3)
            #
            # # Iterate over the contours to find the correct ones
            # for i, cnt in enumerate(contours):
            #     # Calculate the area of the current contour
            #     area = cv2.contourArea(cnt)
            #     valid_area = max_area > area > min_area
            #
            #     # Calculate the fill factor of the contour
            #     filled_contour = np.zeros(img_box.shape[0:2], dtype=np.uint8)
            #     cv2.drawContours(filled_contour, contours, i, 255, thickness=cv2.FILLED)
            #     total_pixels = np.count_nonzero(filled_contour)
            #
            #     # Get the pixels that are both within the contour and in the binary image
            #     valid_fill_ratio = False
            #     if total_pixels > 200:
            #         filled_contour = cv2.bitwise_and(filled_contour, gray_binary)
            #         filled_pixels = np.count_nonzero(filled_contour)
            #         fill_ratio = filled_pixels / total_pixels
            #         valid_fill_ratio = fill_ratio > MIN_FILL_RATIO
            #
            #     # Draw bounding box if area of contour is sufficient
            #     if valid_area and valid_fill_ratio:
            #         box_cnt += 1
            #         x, y, w, h = cv2.boundingRect(cnt)
            #         cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            #         cv2.imshow(f'Filled Contour AND Binary Image ({box_cnt})', filled_contour)
            #
            # # Debugging information
            # if parameters_changed:
            #     print(80 * '=')
            #     print('Debug information:')
            #     print(f'Using image: {img_path}')
            #     print(f'Found {len(contours)} contours in blurred gray-scale.')
            #     contours2, _ = cv2.findContours(gray_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     print(f'Found {len(contours)} contours in binary image.')
            #     print(80 * '=')
            #
            # row2 = np.hstack((
            #     cv2.cvtColor(gray_masked, cv2.COLOR_GRAY2BGR),
            #     cv2.cvtColor(gray_binary, cv2.COLOR_GRAY2BGR),
            #     cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2BGR),
            #     img_box
            # ))
            # res = np.vstack((row1, row2))
            # cv2.imshow('<Original | Original CLAHE | LAB CLAHE | Gray CLAHE> <Skin Masked | Binary | Edges | Boxes>',
            #            res)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        # Prepare region of interest of hand for neuronal network

        cv2.destroyAllWindows()

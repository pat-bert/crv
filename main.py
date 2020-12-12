import multiprocessing as mp
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from skimage import color, segmentation, img_as_ubyte
from skimage.future import graph
from skimage.segmentation import mark_boundaries

# Skin Color Mask
LMIN, LMAX = 0, 255
AMIN, AMAX = 126, 165
BMIN, BMAX = 128, 214

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


def preprocess(file):
    # Get x random image from the set of allowed labels
    img_original = cv2.imread(file)

    # Enhance the contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Transform to other color spaces
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Mask the skin color
    skin_color_mask = cv2.inRange(lab, (LMIN, AMIN, BMIN), (LMAX, AMAX, BMAX))

    # Segmentation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels = segmentation.slic(img_rgb, n_segments=250, start_label=1, slic_zero=True)
    # print(f"SLIC number of segments: {len(np.unique(labels))}")
    out = img_as_ubyte(mark_boundaries(img_rgb.copy(), labels))
    img_segmented = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # Graph merging
    img_rgb_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=skin_color_mask)
    g = graph.rag_mean_color(img_rgb_masked, labels)
    labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False, in_place_merge=True,
                                       merge_func=merge_mean_color, weight_func=_weight_mean_color)
    out = color.label2rgb(labels2, img_rgb_masked, kind='avg', bg_label=0)
    out = img_as_ubyte(mark_boundaries(out, labels2, (0, 0, 0)))
    img_merged = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # Convert to mask and apply to original image
    grayscale = cv2.cvtColor(img_merged, cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(grayscale, 10, 255)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones(5))
    thresh = cv2.dilate(thresh, np.ones(3))
    masked_img = cv2.bitwise_and(img, img, mask=thresh)

    # row1 = np.hstack((img_original, image))
    # row2 = np.hstack((img_segmented, img_merged))
    # row3 = np.hstack((cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), masked_img))
    # block = np.vstack((row1, row2, row3))
    # cv2.imshow('Original and CLAHE enhanced |Segmented and Merged | Binary and Masked', block)
    # cv2.destroyAllWindows()

    # Save images
    new_filepath = Path(file.replace('ressource_rgb', 'ressource_slic'))
    new_filepath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imshow(str(new_filepath), masked_img)
    cv2.waitKey()


if __name__ == '__main__':
    # Get all image file paths in one of the source folders
    image_files = glob(r'D:\Nutzer\Documents\PycharmProjects\crv\ressource_rgb\*\*\*.jp*g')

    pool = mp.Pool(processes=1)
    pool.map(preprocess, image_files)

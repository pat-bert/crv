from collections import defaultdict
from glob import glob
from math import hypot, exp, log
from typing import Optional

import cv2
import numpy as np
from skimage import segmentation
from skimage.measure import regionprops

# noinspection PyProtectedMember

# LAB color quantization levels
N_QUANT = 14
# Number of target superpixels for SLIC
N_SLIC = 150

DEBUG = False


def debug_image(title, img):
    if DEBUG:
        cv2.imshow(title, img)


def calc_skin_prob(img):
    # TODO Calculate skin probability
    prob = np.logical_and.reduce((124 <= img[:, :, 1], img[:, :, 1] <= 165, 126 <= img[:, :, 1], img[:, :, 1] <= 214))
    prob = prob.astype(dtype=np.float)
    return prob


def calc_spatial_distance(reg1: int, reg2: int, shape, centroids) -> float:
    """
    Calculates the spatial distance between two superpixels specified by their labels
    :param shape:
    :param reg1: Label of the first superpixel
    :param reg2: Label of the second superpixel
    :param centroids: Properties of the superpixels
    :return:
    """
    x1, y1 = centroids[reg1 - 1]
    x2, y2 = centroids[reg2 - 1]
    x_max, y_max = shape[0:2]
    spatial_distance = hypot((x1 - x2) / x_max, (y1 - y2) / y_max)
    return spatial_distance


def calc_skin_probability_distance(labels, x: int, y: int, p_skin, areas):
    """
    Skin probability distance between region x and region y defined as:
    D_C = SUM(skin probability of all pixels of region x)/(Number of Pixels in region x)
        - SUM(skin probability of all pixels of region y)/(Number of Pixels in region y)
    :param labels:
    :param x:
    :param y:
    :param p_skin:
    :param areas:
    :return:
    """
    # Calculate the skin color probability distance as difference of the normalized probabilities
    skin_prop_distance = np.sum(p_skin[labels == x]) / areas[x - 1] - np.sum(p_skin[labels == y]) / areas[y - 1]
    return skin_prop_distance


def calc_region_saliency(p_skin, labels, centroids, areas, sigma: Optional[float] = 0.7):
    """
    Calculate x region-level saliency for an image segmented into superpixels

    Salience of region k is defined as:
    S_R(k) = SUM{N_Ri*D_C(R_k,R_j)*exp(-D_s(R_k,R_j)/sigma)} over all regions i, for i!=k
        N_Ri    "Number of pixels in region i"
        D_C     "Skin probability distance between regions"
        D_S     "Spatial distance of region centroids"
        sigma   "Strength of spatial distance weighting"

    Consequently, the saliency of the most aggregate and skin-like regions is enhanced.
    :param p_skin: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param centroids:
    :param areas:
    :param sigma: Strength of spatial distance weighting
    :return:
    """
    if sigma == 0:
        raise ValueError("Sigma may not be zero.")

    # Create empty matrix with shape of image
    saliency = np.zeros(labels.shape)

    # Find unique superpixel labels and iterate over each label
    unique_labels = np.unique(labels)
    for x in unique_labels:
        curr_sal = 0
        for y in unique_labels:
            # Compare current superpixel with all other superpixels
            if x != y:
                # Calculate spatial distance between self and other superpixel's centroid
                d_s = calc_spatial_distance(x, y, labels.shape, centroids)

                # Calculate the skin color probability distance between the superpixels
                y_area = areas[y - 1]
                d_c = calc_skin_probability_distance(labels, x, y, p_skin, areas)

                # Update saliency for current superpixel (minus is not in Article, probably an error)
                curr_sal += y_area * d_c * exp(-d_s / sigma)

        # Set saliency for current superpixel
        saliency[labels == x] = curr_sal

    # Normalize to [0,255]
    saliency = saliency * 255 / np.amax(saliency)
    return saliency.astype(dtype=np.uint8)


def region_color_hist(img, labels):
    """
    Count the pixels for each color combination in the whole image and in the individual regions
    :param img: The image to be used for color counting
    :param labels: Array of labels containing the region a pixel belongs to
    :return:
    """
    total_occurences = defaultdict(lambda: 0)
    region_occurences = defaultdict(lambda: defaultdict(lambda: 0))

    x_max, y_max = img.shape[0:2]
    for x in range(0, x_max):
        for y in range(0, y_max):
            color = tuple(img[x, y])
            label = labels[x, y]
            total_occurences[color] += 1
            region_occurences[label][color] += 1

    return total_occurences, region_occurences


def color_entropy(img, total_occurences, region_occurences, centroids):
    """
    Calculate the color based entropy defined as:
        FOR COLOR i
        SUM OVER ALL REGIONS j
        {- p_ij*log_2(p_ij)*SUMD}
        WITH SUMD = SUM OVER ALL OTHER REGIONS k
        {D_S(j,k)*p_ik}

        p_ij    "Ratio of pixels with color i between in region j and whole image"
        p_ik    "Ratio of pixels with color i between in region k and whole image"
        D_s     "Spatial distance of region centroids"

    :param img:
    :param total_occurences: Dictionary of occurences of a specific color combination
    :param region_occurences: Dictionary of occurences of a specific color combination by regions
    :param centroids: Properties of the superpixels
    :return:
    """
    entropy = defaultdict(lambda: 0)
    # Calculate the entropy for each color in the histogram
    for color in total_occurences.keys():
        # Iterate over all regions
        for j in region_occurences.keys():
            # Skip regions without contribution
            if region_occurences[j][color] == 0:
                continue
            sum_others = 0
            # Iterate over all other regions
            for k in region_occurences.keys():
                # Skip equal regions
                if j != k:
                    # Calculate spatial distance between the regions
                    d_s = calc_spatial_distance(j, k, img.shape, centroids)
                    # Ratio of color occurence in region and total image
                    p_ik = region_occurences[k][color] / total_occurences[color]
                    sum_others += d_s * p_ik

            # Ratio of color occurence in region and total image
            p_ij = region_occurences[j][color] / total_occurences[color]
            if sum_others != 0:
                entropy[color] += -p_ij * log(p_ij, 2) * sum_others

    return entropy


def calc_pixel_saliency(img, p_skin, labels, centroids, sigma: Optional[float] = 0.3):
    """
    Calculate x region-level saliency for an image segmented into superpixels
    Idea:
    - Background colors have broader spatial distribution and more balance distribution among superpixels
    - Foreground colors have x more concentrated distribution
    :param img:
    :param p_skin: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param centroids:
    :param sigma:
    :return:
    """
    if sigma == 0:
        raise ValueError("Sigma may not be zero.")

    # Get overall and region specific color histograms
    total_occurences, region_occurences = region_color_hist(img, labels)

    # Calculate entropy for each three-value color combination
    entropy_by_color = color_entropy(img, total_occurences, region_occurences, centroids)

    # Assign the pixel based entropy for each pixel
    x_max, y_max = img.shape[0:2]
    entropy_by_pixel = np.zeros(img.shape[0:2])
    for x in range(0, x_max):
        for y in range(0, y_max):
            color = tuple(img[x, y])
            entropy_by_pixel[x, y] = entropy_by_color[color]

    # Normalize or the exp(x,sigma) will not make sense!!
    entropy_by_pixel_n = entropy_by_pixel / np.amax(entropy_by_pixel)

    # Calculate pixel based saliency
    m_saliency = np.exp(-entropy_by_pixel_n / sigma)
    saliency = np.multiply(p_skin, np.sqrt(m_saliency))
    # Normalize to [0,255]
    saliency = saliency * 255 / np.amax(saliency)
    saliency = saliency.astype(dtype=np.uint8)
    return saliency


def fuse_saliency_maps(map1, map2):
    """
    Two different saliency maps are combined
    :param map1: First saliency map
    :param map2: Second saliency map
    :return: Fused saliency map, using sqrt(a,b)*a where a is a center-biased weight
    """
    # Check whether the maps are compatible
    if map1.shape != map2.shape:
        raise ValueError("Saliency maps must have equal shape.")

    # Center-bias weight
    width, height = map1.shape
    x, y = np.indices((width, height))
    nominator = -np.hypot(0.5 * (width - 1) - x, 0.5 * (height - 1) - y)
    denominator = (0.5 * min(width, height)) ** 2
    alpha = np.exp(nominator / denominator)

    # Combine the maps and apply the bias
    root = np.sqrt(np.multiply(map1, map2))
    c = np.multiply(alpha, root)
    c = c.astype(dtype=np.uint8)
    return c


if __name__ == '__main__':
    # Get x random image from the set of allowed labels
    image_files = glob(r'D:\Nutzer\Documents\PycharmProjects\crv\ressource_rgb\*\*\*.jp*g')
    image = cv2.imread(np.random.choice(image_files))
    debug_image('Image', image)

    # Transform to other color spaces
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Color quantization
    lab = np.round(lab * (N_QUANT / 255)) * (255 / N_QUANT)
    lab = lab.astype(dtype=np.uint8)

    # Step 1: Segment image using SLIC (k-means clustering in x,y,x,y,luminance)
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    segments = segmentation.slic(img_rgb, n_segments=N_SLIC, start_label=1, slic_zero=True, enforce_connectivity=True)
    regions = regionprops(segments)
    region_centroids = tuple(i.centroid for i in regions)
    region_areas = tuple(i.area for i in regions)

    # Calculate skin color probability for each pixel of the image
    prob_skin = calc_skin_prob(lab)

    # Step 2: Compute pixel-level saliency map
    m_e = calc_pixel_saliency(lab, prob_skin, segments, region_centroids)
    debug_image('Pixel-based Saliency M_e', m_e)

    # Step 3: Compute region-level saliency map
    m_c = calc_region_saliency(prob_skin, segments, region_centroids, region_areas)
    debug_image('Region-based Saliency M_c', m_c)

    # Step 4: Fuse the confidence maps and binarize using Otsu method
    # m_coarse = fuse_saliency_maps(m_e, m_c)
    m_coarse = m_e
    m_coarse = cv2.GaussianBlur(m_coarse, (5, 5), 0)
    _, thresh = cv2.threshold(m_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_image('Coarse Saliency', m_coarse)
    debug_image('Coarse mask', thresh)

    # Step 5: Calculate observation likelihood probability
    mask = cv2.inRange(thresh, 1, 255)
    foreground = cv2.bitwise_and(lab, lab, mask=mask)
    background = cv2.bitwise_and(lab, lab, mask=255 - mask)
    debug_image('Foreground', foreground)
    debug_image('Background', background)
    # TODO Calculate observation likelihood probability
    p_v_fg = np.prod([1])
    p_v_bg = np.prod([1])

    # Step 6: Bayesian framework to obtain fine confidence map and binarize using Otsu method
    m_fine = (np.multiply(m_coarse, p_v_fg)) / (np.multiply(m_coarse, p_v_fg) + np.multiply(1 - m_coarse, p_v_bg))
    m_fine = m_fine.astype(dtype=np.uint8)
    m_fine = cv2.GaussianBlur(m_fine, (5, 5), 0)
    _, m_fine = cv2.threshold(m_fine, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_image('MFINE', m_coarse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

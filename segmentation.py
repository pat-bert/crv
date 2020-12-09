from collections import defaultdict
from glob import glob
from math import hypot, exp, log
from typing import Optional

import cv2
import numpy as np
from skimage import segmentation
from skimage.measure import regionprops


# Algorithm parameters
N_QUANT = 14                # LAB color quantization levels
N_SLIC = 120                # Number of target superpixels for SLIC
FINAL_OPENING_SIZE = 7     # Opening of the final mask
FINAL_CLOSING_SIZE = 11      # Closing of the final mask

DEBUG = True


def debug_image(title, img):
    if DEBUG:
        cv2.imshow(title, img)


def skin_probability(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # prob = np.logical_and.reduce((118 <= img[:, :, 1], img[:, :, 1] <= 165, 118 <= img[:, :, 2], img[:, :, 2] <= 210))
    denominator = np.square(r + g + b)
    prob = np.logical_and.reduce((
        np.divide(r, g) > 1.130,
        np.divide(np.multiply(r, b), denominator) > 0.107,
        np.divide(np.multiply(r, g), denominator) > 0.112,
    ))

    # TODO Create more steps between zero and ones
    prob = 255 * prob.astype(dtype=np.uint8)
    debug_image('Mask', prob)
    prob = prob.astype(dtype=np.float)
    return prob


def all_spatial_distances(labels, centroids, shape):
    unique_labels = np.unique(labels)
    max_label = len(unique_labels)
    distances = defaultdict(lambda: dict())
    x_max, y_max = shape[0:2]

    for i in range(1, max_label + 1):
        for j in range(i + 1, max_label + 1):
            x1, y1 = centroids[i - 1]
            x2, y2 = centroids[j - 1]
            d_s = hypot((x1 - x2) / x_max, (y1 - y2) / y_max)
            distances[i][j] = d_s

    return distances


def region_saliency(p_skin, labels, distances, areas, sigma: Optional[float] = 0.7):
    """
    Calculate x region-level saliency for an image segmented into superpixels

    Salience of region k is defined as:
    S_R(k) = SUM{N_Ri*D_C(R_k,R_j)*exp(-D_s(R_k,R_j)/sigma)} over all regions i, for i!=k
        N_Ri    "Number of pixels in region i"
        D_C     "Skin probability distance between regions"
        D_S     "Spatial distance of region centroids"
        sigma   "Strength of spatial distance weighting"

    Skin probability distance between region x and region y defined as:
    D_C = SUM(skin probability of all pixels of region x)/(Number of Pixels in region x)
        - SUM(skin probability of all pixels of region y)/(Number of Pixels in region y)

    Consequently, the saliency of the most aggregate and skin-like regions is enhanced.
    :param p_skin: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param distances:
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
        d_c_x = np.sum(p_skin[labels == x]) / areas[x - 1]
        for y in unique_labels:
            # Compare current superpixel with all other superpixels
            if x != y:
                # Calculate spatial distance between self and other superpixel's centroid
                d_s = distances[min(x, y)][max(x, y)]

                # Calculate the skin color probability distance between the superpixels
                d_c = d_c_x - np.sum(p_skin[labels == y]) / areas[y - 1]

                # Update saliency for current superpixel (minus is not in Article, probably an error)
                curr_sal += areas[y - 1] * d_c * exp(-d_s / sigma)

        # Set saliency for current superpixel
        saliency[labels == x] = curr_sal

    # Normalize to [0,255]
    saliency = saliency * 255 / np.amax(saliency)
    return saliency.astype(dtype=np.uint8)


def region_color_hist(img, labels=None):
    """
    Count the pixels for each color combination in the whole image and in the individual regions
    :param img: The image to be used for color counting
    :param labels: Array of labels containing the region a pixel belongs to
    :return:
    """
    total_occurences = defaultdict(lambda: 0)

    x_max, y_max = img.shape[0:2]
    if labels is not None:
        region_occurences = defaultdict(lambda: defaultdict(lambda: 0))
        for x in range(0, x_max):
            for y in range(0, y_max):
                color = tuple(img[x, y])
                label = labels[x, y]
                total_occurences[color] += 1
                region_occurences[label][color] += 1
    else:
        region_occurences = None
        for x in range(0, x_max):
            for y in range(0, y_max):
                color = tuple(img[x, y])
                total_occurences[color] += 1

    return total_occurences, region_occurences


def color_entropy(total_occurences, region_occurences, distances):
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

    :param total_occurences: Dictionary of occurences of a specific color combination
    :param region_occurences: Dictionary of occurences of a specific color combination by regions
    :param distances: Properties of the superpixels
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

            # Iterate over all other regions
            sum_others = 0
            for k in region_occurences.keys():
                # Skip equal regions
                if j != k:
                    # Calculate spatial distance between the regions
                    d_s = distances[min(j, k)][max(j, k)]
                    # Ratio of color occurence in region and total image
                    p_ik = region_occurences[k][color] / total_occurences[color]
                    sum_others += d_s * p_ik

            # Ratio of color occurence in region and total image
            p_ij = region_occurences[j][color] / total_occurences[color]
            if sum_others != 0:
                entropy[color] += -p_ij * log(p_ij, 2) * sum_others

    return entropy


def pixel_saliency(img, p_skin, labels, distances, sigma: Optional[float] = 0.3):
    """
    Calculate x region-level saliency for an image segmented into superpixels
    Idea:
    - Background colors have broader spatial distribution and more balance distribution among superpixels
    - Foreground colors have x more concentrated distribution
    :param img:
    :param p_skin: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param distances:
    :param sigma:
    :return:
    """
    if sigma == 0:
        raise ValueError("Sigma may not be zero.")

    # Get overall and region specific color histograms
    total_occurences, region_occurences = region_color_hist(img, labels)

    # Calculate entropy for each three-value color combination
    entropy_by_color = color_entropy(total_occurences, region_occurences, distances)

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


def likelihood_probability(foreground, background):
    total_occurances_fg, _ = region_color_hist(foreground)
    total_occurances_bg, _ = region_color_hist(background)
    n_fg = np.count_nonzero(foreground)
    n_bg = np.count_nonzero(background)
    p_v_fg = np.zeros(lab.shape[0:2])
    p_v_bg = np.zeros(lab.shape[0:2])
    for row in range(0, lab.shape[0]):
        for col in range(0, lab.shape[1]):
            pixel_color = tuple(lab[row, col])
            p_v_fg[row, col] = total_occurances_fg[pixel_color] / n_fg
            p_v_bg[row, col] = total_occurances_bg[pixel_color] / n_bg

    return p_v_fg, p_v_bg


if __name__ == '__main__':
    while True:
        # Get x random image from the set of allowed labels
        image_files = glob(r'D:\Nutzer\Documents\PycharmProjects\crv\ressource_rgb\*\*\*.jp*g')
        image_path = np.random.choice(image_files)
        print(f'Using image: {image_path}')
        image_path = r'D:\Nutzer\Documents\PycharmProjects\crv\ressource_rgb\Training\25\Subject46_Scene3_rgb1_000427.jpg'
        image = cv2.imread(image_path)
        debug_image('Image', image)

        # Transform to other color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Color quantization
        lab = np.round(lab * (N_QUANT / 255)) * (255 / N_QUANT)
        lab = lab.astype(dtype=np.uint8)

        # Step 1: Segment image using SLIC (k-means clustering in x,y,x,y,luminance)
        img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        segments = segmentation.slic(img_rgb, n_segments=N_SLIC, start_label=1, slic_zero=True,
                                     enforce_connectivity=True)
        regions = regionprops(segments)
        region_centroids = tuple(i.centroid for i in regions)
        region_areas = tuple(i.area for i in regions)
        all_distances = all_spatial_distances(segments, region_centroids, lab.shape)

        # Calculate skin color probability for each pixel of the image
        prob_skin = skin_probability(image)

        # Step 2: Compute pixel-level saliency map
        m_e = pixel_saliency(lab, prob_skin, segments, all_distances)
        debug_image('Pixel-based Saliency M_e', m_e)

        # Step 3: Compute region-level saliency map
        m_c = region_saliency(prob_skin, segments, all_distances, region_areas)
        debug_image('Region-based Saliency M_c', m_c)

        # Step 4: Fuse the confidence maps and binarize using Otsu method
        # m_coarse = fuse_saliency_maps(m_e, m_c)
        m_coarse = m_e
        m_coarse = cv2.GaussianBlur(m_coarse, (5, 5), 0)
        _, thresh_coarse = cv2.threshold(m_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        debug_image('Coarse Saliency', m_coarse)
        debug_image('Coarse mask', thresh_coarse)
        mask = cv2.inRange(thresh_coarse, 1, 255)
        coarse_foreground = cv2.bitwise_and(lab, lab, mask=mask)
        coarse_background = cv2.bitwise_and(lab, lab, mask=255 - mask)

        # Step 5: Calculate observation likelihood probability
        p_v_fg, p_v_bg = likelihood_probability(coarse_foreground, coarse_background)

        # Step 6: Bayesian framework to obtain fine confidence map and binarize using Otsu method
        m_fine = 255 * np.divide(np.multiply(m_coarse, p_v_fg),
                                 np.multiply(m_coarse, p_v_fg) + np.multiply(1 - m_coarse, p_v_bg))
        m_fine = m_fine.astype(dtype=np.uint8)
        m_fine = cv2.GaussianBlur(m_fine, (5, 5), 0)
        _, thresh_fine = cv2.threshold(m_fine, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        debug_image('MFINE', m_fine)

        # Final touches to the mask
        thresh_fine = cv2.morphologyEx(thresh_fine, cv2.MORPH_OPEN, np.ones((FINAL_OPENING_SIZE, FINAL_OPENING_SIZE)))
        thresh_fine = cv2.morphologyEx(thresh_fine, cv2.MORPH_CLOSE, np.ones((FINAL_CLOSING_SIZE, FINAL_CLOSING_SIZE)))

        # Apply the mask
        mask = cv2.inRange(thresh_fine, 1, 255)
        fine_foreground = cv2.bitwise_and(image, image, mask=mask)
        fine_background = cv2.bitwise_and(image, image, mask=255 - mask)

        debug_image('Foreground', fine_foreground)
        debug_image('Background', fine_background)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

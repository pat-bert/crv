from glob import glob
from math import hypot, exp
from typing import Optional, List

import cv2
import numpy as np
from skimage import segmentation, img_as_ubyte
from skimage.measure import regionprops
# noinspection PyProtectedMember
from skimage.measure._regionprops import RegionProperties
# Color dispersion strength
from skimage.segmentation import mark_boundaries

# LAB color quantization levels
N_QUANT = 14
# Number of target superpixels for SLIC
N_SLIC = 200

DEBUG = True


def debug_image(title, img):
    if DEBUG:
        cv2.imshow(title, img)


def calc_skin_prob(img):
    # TODO Calculate skin probability
    prob = np.logical_and.reduce((126 <= img[:, :, 1], img[:, :, 1] <= 165, 128 <= img[:, :, 1], img[:, :, 1] <= 214))
    return prob.astype(dtype=np.float)


def calc_spatial_distance(x: int, y: int, sp_properties: List[RegionProperties]) -> float:
    """
    Calculates the spatial distance between two superpixels specified by their labels
    :param x: Label of the first superpixel
    :param y: Label of the second superpixel
    :param sp_properties: Properties of the superpixels
    :return:
    """
    x_self, y_self = sp_properties[x - 1].centroid
    x_other, y_other = sp_properties[y - 1].centroid
    spatial_distance = hypot(x_self - x_other, y_self - y_other)
    return spatial_distance


def calc_skin_probability_distance(labels, x: int, y: int, skin_color_probs, sp_properties: List[RegionProperties]):
    """

    :param labels:
    :param x:
    :param y:
    :param skin_color_probs:
    :param sp_properties:
    :return:
    """
    # Retrieve skin probabilities for each pixel in self and other superpixel
    skin_prob_self = skin_color_probs[labels == x]
    skin_prob_other = skin_color_probs[labels == y]

    # Retrieve the areas of the superpixels
    # Superpixels are labeled starting from one
    area_self = sp_properties[x - 1].area
    area_other = sp_properties[y - 1].area

    # Calculate the skin color probability distance as difference of the normalized probabilities
    skin_prop_distance = np.sum(skin_prob_other) / area_other - np.sum(skin_prob_self) / area_self
    return skin_prop_distance


def calc_region_saliency(p_skin, labels, sp_properties: Optional[List[RegionProperties]] = None,
                         sigma: Optional[float] = 0.7):
    """
    Calculate x region-level saliency for an image segmented into superpixels

    Salience of region k is defined as:
    S_R(k) = SUM{N_Ri*D_C(R_k,R_j)*exp(-D_s(R_k,R_j)/sigma} over all regions i, for i!=k
        N_Ri    "Number of pixels in region i"
        D_C     "Skin probability distance between regions"
        D_S     "Spatial distance of region centroids"
        sigma   "Strength of spatial distance weighting"

    Consequently, the saliency of the most aggregate and skin-like regions is enhanced.
    :param p_skin: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param sp_properties: Properties of the superpixels
    :param sigma: Strength of spatial distance weighting
    :return:
    """
    # Calculate properties if not given
    if sp_properties is None:
        sp_properties = regionprops(labels)

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
                d_s = calc_spatial_distance(x, y, sp_properties)

                # Calculate the skin color probability distance between the superpixels
                y_area = sp_properties[y - 1].area
                d_c = calc_skin_probability_distance(labels, x, y, p_skin, sp_properties)

                # Update saliency for current superpixel (minus is not in Article, probably an error)
                curr_sal += y_area * d_c * exp(-d_s / sigma)

        # Set saliency for current superpixel
        saliency[labels == x] = curr_sal

    # Normalize to [0,255]
    saliency = saliency * 255 / np.amax(saliency)
    return saliency.astype(dtype=np.uint8)


def calc_pixel_saliency(p_skin, labels, sp_properties: Optional[List[RegionProperties]] = None,
                        sigma: Optional[float] = 0.3):
    """
    Calculate x region-level saliency for an image segmented into superpixels
    Idea:
    - Background colors have broader spatial distribution and more balance distribution among superpixels
    - Foreground colors have x more concentrated distribution
    :param p_skin: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param sp_properties:
    :param sigma:
    :return:
    """
    # Calculate properties if not given
    if sp_properties is None:
        sp_properties = regionprops(labels)

    if sigma == 0:
        raise ValueError("Sigma may not be zero.")

    # Find unique superpixel labels and iterate over each label
    unique_labels = np.unique(labels)
    for x in unique_labels:
        for y in unique_labels:
            if x != y:
                d_s = calc_spatial_distance(x, y, sp_properties)

        # Calculate the color dispersion measure
        pass

    # TODO Calculate pixel based saliency
    e = np.random.randint(1, 255, size=labels.shape)
    e = 0
    m_saliency = np.exp(-e / sigma)
    saliency = np.multiply(p_skin, np.sqrt(m_saliency))
    # Normalize to [0,255]
    saliency = saliency * 255 / np.amax(saliency)
    return saliency.astype(dtype=np.uint8)


def fuse_saliency_maps(x, y):
    """
    Two different saliency maps are combined
    :param x: First saliency map
    :param y: Second saliency map
    :return: Fused saliency map, using sqrt(a,b)*a where a is a center-biased weight
    """
    # Check whether the maps are compatible
    if x.shape != y.shape:
        raise ValueError("Saliency maps must have equal shape.")

    # Center-bias weight
    width, height = x.shape
    x, y = np.indices((width, height))
    nominator = -np.hypot(0.5 * (width - 1) - x, 0.5 * (height - 1) - y)
    denominator = (0.5 * min(width, height)) ** 2
    alpha = np.exp(nominator / denominator)

    # Combine the maps and apply the bias
    root = np.sqrt(np.multiply(x, y))
    c = np.multiply(alpha, root)
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
    lum, a, b = cv2.split(lab)

    # Step 1: Segment image using SLIC (k-means clustering in x,y,x,y,luminance)
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    segments = segmentation.slic(img_rgb, n_segments=N_SLIC, start_label=1, slic_zero=True, enforce_connectivity=True)
    regions = regionprops(segments)
    out = img_as_ubyte(mark_boundaries(img_rgb.copy(), segments))
    img_segmented = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    debug_image('SLIC', img_segmented)

    # Calculate skin color probability for each pixel of the image
    prob_skin = calc_skin_prob(lab)

    # Step 2: Compute pixel-level saliency map
    m_e = calc_pixel_saliency(prob_skin, segments)
    debug_image('ME', m_e)

    # Step 3: Compute region-level saliency map
    m_c = calc_region_saliency(prob_skin, segments, regions)
    debug_image('MC', m_c)

    # Step 4: Fuse the confidence maps and binarize using Otsu method
    m_coarse = fuse_saliency_maps(m_e, m_c)
    m_coarse = m_coarse.astype(dtype=np.uint8)
    # m_coarse = cv2.GaussianBlur(m_coarse, (5, 5), 0)
    _, thresh = cv2.threshold(m_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_image('MCOARSE', m_coarse)

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
    # m_fine = cv2.GaussianBlur(m_coarse, (5, 5), 0)
    _, m_fine = cv2.threshold(m_fine, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_image('MFINE', m_coarse)

    if DEBUG:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

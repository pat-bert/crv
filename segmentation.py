from glob import glob
from math import hypot, exp
from typing import Optional, List

import cv2
import numpy as np
from skimage import segmentation
from skimage.measure import regionprops
# noinspection PyProtectedMember
from skimage.measure._regionprops import RegionProperties

# Color dispersion strength
SIGMA_E = 0.3
# Spatial distance weighing strength
SIGMA_S = 0.7
# LAB color quantization levels
N_QUANT = 12


def calc_skin_prob(img):
    # TODO Calculate skin probability
    return np.random.random(img.shape[0:2])


def calc_superpixel_distance(x: int, y: int, superpixel_props: List[RegionProperties]) -> float:
    """
    Calculates the spatial distance between two superpixels specified by their labels
    :param x: Label of the first superpixel
    :param y: Label of the second superpixel
    :param superpixel_props: Properties of the superpixels
    :return:
    """
    # Superpixels are labeled starting from one
    x_self, y_self = superpixel_props[x - 1].centroid
    x_other, y_other = superpixel_props[y - 1].centroid
    spatial_distance = hypot(x_self - x_other, y_self - y_other)
    return spatial_distance


def calc_region_saliency(skin_color_probs, labels, superpixel_props: Optional[List[RegionProperties]] = None,
                         sigma: Optional[float] = SIGMA_S):
    """
    Calculate a region-level saliency for an image segmented into superpixels
    :param skin_color_probs: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param superpixel_props: Properties of the superpixels
    :param sigma:
    :return:
    """
    # Calculate properties if not given
    if superpixel_props is None:
        superpixel_props = regionprops(labels)

    if sigma == 0:
        raise ValueError("Sigma may not be zero.")

    # Create empty matrix with shape of image
    saliency = np.zeros(labels.shape)

    # Find unique superpixel labels and iterate over each label
    unique_labels = np.unique(labels)
    for self_label in unique_labels:
        curr_sal = 0
        for other_label in unique_labels:
            # Compare current superpixel with all other superpixels
            if self_label != other_label:
                # Calculate spatial distance between self and other superpixel's centroid
                spatial_distance = calc_superpixel_distance(self_label, other_label, superpixel_props)

                # Retrieve skin probabilities for each pixel in self and other superpixel
                skin_prob_self = skin_color_probs[labels == self_label]
                skin_prob_other = skin_color_probs[labels == other_label]

                # Retrieve the areas of the superpixels
                # Superpixels are labeled starting from one
                area_self = superpixel_props[self_label - 1].area
                area_other = superpixel_props[other_label - 1].area

                # Calculate the skin color probability distance as difference of the normalized probabilities
                dc = np.sum(skin_prob_self) / area_self - np.sum(skin_prob_other) / area_other

                # Update saliency for current superpixel
                curr_sal += superpixel_props[self_label - 1].area * dc * exp(spatial_distance / sigma)

        # Set saliency for current superpixel
        saliency[labels == self_label] = curr_sal

    # Normalize to [0,255]
    saliency = saliency * 255 / np.amax(saliency)
    return saliency.astype(dtype=np.uint8)


def calc_pixel_saliency(skin_color_probs, labels, superpixel_props: Optional[List[RegionProperties]] = None,
                        sigma: Optional[float] = SIGMA_E):
    """
    Calculate a region-level saliency for an image segmented into superpixels
    :param skin_color_probs: Skin color probability for each image
    :param labels: Labels specifying superpixel affiliation for individual pixels in image
    :param superpixel_props:
    :param sigma:
    :return:
    """
    # Calculate properties if not given
    if superpixel_props is None:
        superpixel_props = regionprops(labels)

    if sigma == 0:
        raise ValueError("Sigma may not be zero.")

    # Find unique superpixel labels and iterate over each label
    unique_labels = np.unique(labels)
    for self_label in unique_labels:
        for other_label in unique_labels:
            if self_label != other_label:
                spatial_distance = calc_superpixel_distance(self_label, other_label, superpixel_props)

        # Calculate the color dispersion measure
        pass

    # TODO Calculate pixel based saliency
    e = np.random.randint(1, 255, size=labels.shape)
    m_saliency = np.exp(-e / sigma)
    saliency = np.multiply(skin_color_probs, np.sqrt(m_saliency))
    # Normalize to [0,255]
    saliency = saliency * 255 / np.amax(saliency)
    return saliency.astype(dtype=np.uint8)


def fuse_saliency_maps(a, b):
    """
    Two different saliency maps are combined
    :param a: First saliency map
    :param b: Second saliency map
    :return: Fused saliency map, using sqrt(a,b)*alpha where alpha is a center-biased weight
    """
    # Check whether the maps are compatible
    if a.shape != b.shape:
        raise ValueError("Saliency maps must have equal shape.")

    # Center-bias weight
    width, height = a.shape
    x, y = np.indices((width, height))
    nominator = -np.hypot(0.5 * (width - 1) - x, 0.5 * (height - 1) - y)
    denominator = (0.5 * min(width, height)) ** 2
    alpha = np.exp(nominator / denominator)

    # Combine the maps and apply the bias
    root = np.sqrt(np.multiply(a, b))
    c = np.multiply(alpha, root)
    return c


if __name__ == '__main__':
    # Get a random image from the set of allowed labels
    image_files = glob(r'D:\Nutzer\Documents\PycharmProjects\crv\ressource_rgb\*\*\*.jp*g')
    image = cv2.imread(np.random.choice(image_files))
    cv2.imshow('Image', image)

    # Transform to other color spaces
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Color quantization
    lab = np.round(lab * (N_QUANT / 255)) * (255 / N_QUANT)
    lab = lab.astype(dtype=np.uint8)

    # Step 1: Segment image using SLIC
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    segments = segmentation.slic(img_rgb, n_segments=100, start_label=1, slic_zero=True, enforce_connectivity=True)
    regions = regionprops(segments)

    # Calculate skin color probability for each pixel of the image
    skin_color_probabilities = calc_skin_prob(lab)

    # Step 2: Compute pixel-level saliency map
    m_e = calc_pixel_saliency(skin_color_probabilities, segments)
    cv2.imshow('ME', m_e)

    # Step 3: Compute region-level saliency map
    m_c = calc_region_saliency(skin_color_probabilities, segments, regions)
    cv2.imshow('MC', m_c)

    # Step 4: Fuse the confidence maps and binarize using Otsu method
    m_coarse = fuse_saliency_maps(m_e, m_c)
    m_coarse = m_coarse.astype(dtype=np.uint8)
    # m_coarse = cv2.GaussianBlur(m_coarse, (5, 5), 0)
    _, m_coarse = cv2.threshold(m_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('MCOARSE', m_coarse)

    # Step 5: Calculate observation likelihood probability
    # Set to invariant elements for now
    p_v_fg = 1
    p_v_bg = 1

    # Step 6: Bayesian framework to obtain fine confidence map and binarize using Otsu method
    m_fine = (m_coarse * p_v_fg) / (m_coarse * p_v_fg + (1 - m_coarse) * p_v_bg)
    m_fine = m_fine.astype(dtype=np.uint8)
    # m_fine = cv2.GaussianBlur(m_coarse, (5, 5), 0)
    _, m_fine = cv2.threshold(m_fine, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('MFINE', m_coarse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

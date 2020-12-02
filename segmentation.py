from glob import glob
from math import hypot, exp

import cv2
import numpy as np
from skimage import segmentation
from skimage.measure import regionprops

SIGMA_E = 0.3
SIGMA_S = 0.7


def calc_skin_prob(img):
    # TODO Calculate skin probability
    return np.random.random(img.shape)


def calc_region_saliency(skin_color_probs, labels, superpixel_props=None, sigma=SIGMA_S):
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
                # Superpixels are labeled starting from one
                x_self, y_self = superpixel_props[self_label - 1].centroid
                x_other, y_other = superpixel_props[other_label - 1].centroid
                spatial_distance = hypot(x_self - x_other, y_self - y_other)

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


def calc_pixel_saliency(labels, sigma=SIGMA_E):
    # TODO Calculate pixel based saliency
    E = np.random.randint(1, 255, size=labels.shape)
    m_saliency = np.exp(-E / sigma)
    P_skin = 1
    saliency = P_skin * np.sqrt(m_saliency)
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

    # Step 1: Segment image using SLIC
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = segmentation.slic(img_rgb, n_segments=100, start_label=1, slic_zero=True, enforce_connectivity=True)
    regions = regionprops(segments)

    # Calculate skin color probability for each pixel of the image
    skin_color_probabilities = calc_skin_prob(image)

    # Step 2: Compute pixel-level saliency map
    m_e = calc_pixel_saliency(segments)
    cv2.imshow('ME', m_e)

    # Step 3: Compute region-level saliency map
    m_c = calc_region_saliency(skin_color_probabilities, segments, regions)
    cv2.imshow('MC', m_c)

    # Step 4: Fuse the confidence maps
    m_coarse = fuse_saliency_maps(m_e, m_c)
    m_coarse = m_coarse.astype(dtype=np.uint8)
    cv2.imshow('MCOARSE', m_coarse)

    # Step 5: Calculate observation likelihood probability
    p_v_fg = 1
    p_v_bg = 1

    # Step 6: Bayesian framework to obtain fine confidence map and binarize using Otsu method
    m_fine = (m_coarse * p_v_fg) / (m_coarse * p_v_fg + (1 - m_coarse) * p_v_bg)
    # m_fine = cv2.GaussianBlur(m_coarse, (5, 5), 0)
    ret, m_fine = cv2.threshold(m_fine, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('MFINE', m_coarse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

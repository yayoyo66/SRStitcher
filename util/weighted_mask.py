import cv2
import numpy as np
from PIL import Image


def make_mask(mask1, mask2, K, sigma1, sigma2):
    content_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones(K, np.float32)

    dilated_mask1= cv2.dilate(mask1, kernel, iterations=1)
    eroded_mask1 = cv2.erode(mask1, kernel, iterations=1)

    tmp_mask1 = cv2.bitwise_xor(dilated_mask1, mask1)
    tmp_mask2 = cv2.bitwise_xor(eroded_mask1, mask1)
    tmp_mask3 = cv2.bitwise_or(tmp_mask1, tmp_mask2)
    seam_mask = cv2.bitwise_and(tmp_mask3, mask2)

    content_mask = cv2.bitwise_not(content_mask)

    gradient = cv2.distanceTransform(seam_mask, cv2.DIST_L1, 3)
    gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
    gradient = (gradient * 255).astype(np.uint8)

    dist_transform2 = cv2.distanceTransform(content_mask, cv2.DIST_L1, 3)
    weight_map2 = cv2.convertScaleAbs(dist_transform2)
    weight_map2 = cv2.normalize(weight_map2, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    max_dist = np.max(gradient)

    scaled_dist_transform = (gradient / max_dist) * sigma1
    scaled_dist_transform = np.uint8(scaled_dist_transform)

    scaled_dist_transform2 = (weight_map2 / max_dist) * sigma2
    scaled_dist_transform2 = np.uint8(scaled_dist_transform2)

    map_result = cv2.bitwise_or(content_mask, gradient)
    map_result2 = cv2.bitwise_xor(scaled_dist_transform, scaled_dist_transform2)
    map_result3 = cv2.bitwise_xor(gradient, weight_map2)

    return Image.fromarray(cv2.bitwise_not(map_result)), content_mask, map_result2, Image.fromarray(cv2.bitwise_not(map_result3))
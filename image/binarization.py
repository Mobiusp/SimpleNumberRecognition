# -*- coding: UTF-8 -*-
"""
@Project ：NumberRecognition
@File    ：binarication.py
@Author  ：Mobiusp

This module is used to preprocess the input image.
"""

import numpy as np
from PIL import Image


def otsu(image: np.ndarray) -> int:
    """
    TODO: Find the boundary between foreground and background
    """
    img_sum = image.shape[0] * image.shape[1]
    his, bins = np.histogram(image, np.arange(0, 257))
    intensity = np.arange(256)
    max_value, res = -1, 0
    for i in bins[1:-1]:
        back = np.sum(his[:i])
        front = np.sum(his[i:])
        w_b = back / img_sum * 1.0
        w_f = front / img_sum * 1.0
        if back == 0 or front == 0:
            continue
        mu_b = np.sum(intensity[:i] * his[:i]) / float(back)
        mu_f = np.sum(intensity[i:] * his[i:]) / float(front)
        value = w_b * w_f * (mu_b - mu_f) ** 2
        if max_value < value:
            max_value = value
            res = i
    return res


def binarization(image_path: str) -> np.ndarray:
    """
    TODO: Binarization of the input image
    """
    img = Image.open(image_path).resize((28, 28), Image.LANCZOS).convert('L')
    image = np.array(img)
    value = otsu(image)
    image = image.astype(np.int16)
    image[image > value] = 256
    image[(image <= value)] = 255 - image[(image <= value)]
    image[image == 256] = 0
    image = image.astype(np.uint8)
    return image.reshape((1, image.size))



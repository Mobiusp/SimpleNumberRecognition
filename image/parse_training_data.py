# -*- coding: UTF-8 -*-
"""
@Project ：NumberRecognition
@File    ：parse_training_data.py
@Author  ：Mobiusp

This module is used to parse MNIST.
MNIST: http://yann.lecun.com/exdb/mnist/
"""

import struct
import numpy as np

# The path of MNIST
PATH = "./training_data"
TRAIN_IMAGE = "/train-images.idx3-ubyte"
TRAIN_LABEL = "/train-labels.idx1-ubyte"


def parse_image(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as file:
        data = file.read()
    magic, num_images, num_rows, num_columns = struct.unpack_from(">iiii", data, 0)
    size = num_images * num_rows * num_columns
    return np.reshape(struct.unpack_from(">" + str(size) + "B", data, struct.calcsize(">iiii")), [num_images, num_rows * num_columns])


def parse_label(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as file:
        data = file.read()
    magic, num_images = struct.unpack_from(">ii", data, 0)
    return np.reshape(struct.unpack_from(">" + str(num_images) + "B", data, struct.calcsize(">ii")), [num_images, 1])


def parse_training_data() -> np.ndarray:
    images = parse_image(PATH + TRAIN_IMAGE)
    labels = parse_label(PATH + TRAIN_LABEL)
    return np.append(images, labels, axis=1)


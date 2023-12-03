# -*- coding: UTF-8 -*-
"""
@Project ：NumberRecognition
@File    ：number_recognition.py
@Author  ：Mobiusp

Use this module to recognition
"""

from model import Model
from image.binarization import binarization
import numpy as np

if __name__ == "__main__":
    image_path = r""
    data = binarization(image_path)

    model = Model(is_training=False, file_path="./number0.01_60_3.00_98.39%_model.model")
    prob = model.forward(data)
    result = int(np.argmax(prob))
    print(f"There is a {float(prob[0][result]) * 100:.6f}% chance that the result will be {result}.")


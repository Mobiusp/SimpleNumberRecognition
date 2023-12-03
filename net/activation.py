# -*- coding: UTF-8 -*-
"""
@Project ：NumberRecognition
@File    ：activation.py
@Author  ：Mobiusp

The Activation Layer.
"""

import numpy as np


class ReLU:
    def __init__(self):
        self.input_x = None

    def forward(self, input_x: np.ndarray):
        self.input_x = input_x
        return np.maximum(0, input_x)

    def backward(self, diff: np.ndarray):
        temp = diff
        temp[self.input_x < 0] = 0
        return temp


class Softmax:
    def __init__(self):
        self.prob = None
        self.loss = -1.0

    def forward(self, input_x: np.ndarray):
        """
        :return: The probability of each number
        """
        temp = np.exp(input_x - np.max(input_x, axis=1, keepdims=True))
        self.prob = temp / np.sum(temp, axis=1, keepdims=True)
        return self.prob

    def backward(self, data: np.ndarray):
        """
        :return: loss
        """
        onehot = np.zeros_like(self.prob)
        onehot[np.arange(self.prob.shape[0]), data] = 1.0
        self.loss = - np.sum(np.log(self.prob) * onehot) / self.prob.shape[0]
        return (self.prob - onehot) / self.prob.shape[0]



# -*- coding: UTF-8 -*-
"""
@Project ：NumberRecognition
@File    ：linearlayer.py
@Author  ：Mobiusp

The Linear Layer.
"""

import numpy as np


class BaseLinearLayer:
    """
    Only used for predicting
    """
    def __init__(self, load_w: np.ndarray, load_b: np.ndarray):
        self.w = load_w
        self.b = load_b
        self.input_x = None

    def forward(self, input_x: np.ndarray) -> np.ndarray:
        self.input_x = input_x
        return np.matmul(input_x, self.w) + self.b


class LinearLayer(BaseLinearLayer):
    def __init__(self, input_num: int, output_num: int, leaning_rate: float = 0.01):
        """
        TODO: Random initialization model params
        """
        super().__init__(np.random.normal(0.0, 0.01, (input_num, output_num)), np.zeros([1, output_num]))
        self.update_w = None
        self.update_b = None
        self.learning_rate = leaning_rate

    def backward(self, diff: np.ndarray) -> np.ndarray:
        self.update_w = np.matmul(self.input_x.T, diff)
        self.update_b = np.sum(diff, axis=0)
        return np.matmul(diff, self.w.T)

    def update(self) -> None:
        self.w -= self.update_w * self.learning_rate
        self.b -= self.update_b * self.learning_rate


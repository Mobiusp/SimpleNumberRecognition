# -*- coding: UTF-8 -*-
"""
@Project ：NumberRecognition
@File    ：model.py
@Author  ：Mobiusp

The number recognition model
"""

import pickle
import numpy as np

from net import *


class Model:
    def __init__(self, is_training: bool = True, file_path: str = "", learning_rate: float = 0.01):
        self.is_training = is_training
        if is_training:
            self.learning_rate = learning_rate
            self.layers = [LinearLayer(784, 256, learning_rate), ReLU(),
                           LinearLayer(256, 128, learning_rate), ReLU(),
                           LinearLayer(128, 64, learning_rate), ReLU(),
                           LinearLayer(64, 10, learning_rate), Softmax()]
        else:
            with open(file_path, "rb") as file:
                model_param = pickle.load(file)
            self.layers = [BaseLinearLayer(model_param[0][0], model_param[0][1]), ReLU(),
                           BaseLinearLayer(model_param[1][0], model_param[1][1]), ReLU(),
                           BaseLinearLayer(model_param[2][0], model_param[2][1]), ReLU(),
                           BaseLinearLayer(model_param[3][0], model_param[3][1]), Softmax()]

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        temp = input_data
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp

    def backward(self, data: np.ndarray) -> float:
        assert self.is_training, "The model is not in training, you can only use it to predict."
        diff = data
        for layer in reversed(self.layers):
            diff = layer.backward(diff)
        return self.layers[len(self.layers) - 1].loss

    def update(self) -> None:
        assert self.is_training, "The model is not in training, you can only use it to predict."
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer.update()

    def save(self, file_name: str):
        assert self.is_training, "The model is not in training, you can only use it to predict."
        temp = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                temp.append((layer.w, layer.b))
        with open(file_name, "wb") as file:
            pickle.dump(temp, file)

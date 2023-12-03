# -*- coding: UTF-8 -*-
"""
@Project ：NumberRecognition
@File    ：training.py
@Author  ：Mobiusp

Use this module to training model.
"""

from model import Model
from image import parse_training_data as ptd
import numpy as np


if __name__ == "__main__":
    model = Model(learning_rate=0.01)
    data = ptd.parse_training_data()

    epoch_times = 100
    batch_size = 60
    batch_times = int(data.shape[0] / batch_size)
    loss = 0.0
    for epoch_time in range(epoch_times):
        np.random.shuffle(data)
        for batch_time in range(batch_times):
            batch_images = data[batch_time * batch_size:(batch_time + 1) * batch_size, :-1]
            batch_labels = data[batch_time * batch_size:(batch_time + 1) * batch_size, -1]
            prob = model.forward(batch_images)
            loss = model.backward(batch_labels)
            model.update()
        print(f"EpochTime: {epoch_time} Loss: {loss:.6f}")
        if loss <= 0.000001:
            break

    test_images = ptd.parse_image(r"./training_data/t10k-images.idx3-ubyte")
    test_labels = ptd.parse_label(r"./training_data/t10k-labels.idx1-ubyte")
    cnt = 0
    for test_index in range(test_images.shape[0]):
        result = np.argmax(model.forward(np.reshape(test_images[test_index], (1, test_images[test_index].shape[0]))))
        cnt += 1 if result == test_labels[test_index] else 0
    print(f"Accuracy: {cnt/ test_images.shape[0] * 100}%")
    model.save(f"number{model.learning_rate}_{batch_size}_{loss*10000000:.2f}_{cnt/ test_images.shape[0] * 100}%_model.model")




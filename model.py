"""
This script contains all the functions related to the model
"""

import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Dropout
from game import MOVES_POSSIBLE, ALL_BLOCK_POSSIBLE, GRID_SIZE_X, GRID_SIZE_Y

EPS: float = 0.4  # probability of playing a random move
# list of all actions possible for the model
LIST_ACTIONS: [int] = [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]


class Model2048(Sequential):
    """
    Create the main model for 2048
    """

    def __init__(self):
        super().__init__()

        # create the model
        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE * MOVES_POSSIBLE,
                   (2, 2),
                   padding="same",
                   input_shape=(GRID_SIZE_Y, GRID_SIZE_X, ALL_BLOCK_POSSIBLE),
                   )
        )
        self.add(LeakyReLU())

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE,  # * MOVES_POSSIBLE
                   (2, 2),
                   padding="same",
                   input_shape=(GRID_SIZE_Y, GRID_SIZE_X, ALL_BLOCK_POSSIBLE),
                   )
        )
        self.add(LeakyReLU())

        self.add(Flatten())

        self.add(Dropout(0.2))

        self.add(Dense(256))
        self.add(LeakyReLU())

        self.add(Dropout(0.2))

        self.add(Dense(4, activation="softmax"))

    def save_model(self, path: str) -> None:
        """
        This function save the model as a h5 file
        :param path: the path where to save the model
        :return: None
        """
        self.save(path)

    def load_performance(self, path) -> None:
        """
        This function will load the weight of the model
        it is better than a tf.keras.load_model
        :param path: the path of the model to load
        :return: None
        """
        _m = Model2048()
        _m = tf.keras.models.load_model(path)
        _w = m.get_weights()
        self.set_weights(_w)

    def take_action(self, grid):
        """
        This function will take sometime a random action and sometime a model action
        :param grid: a sequence of indicators of length SEQUENCE_LENGTH
        :return: the action take
        """
        if random.uniform(0, 1) < EPS:
            # take random action
            action = random.choice(LIST_ACTIONS)

        else:
            # let model choose a action
            action = self.predict(np.array([grid]))[0]

            returned_list = [0, 0, 0, 0]
            returned_list[np.argmax(action)] = 1

        return action


if __name__ == '__main__':
    m = Model2048()

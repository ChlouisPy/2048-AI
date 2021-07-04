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

MUTATION_RATE: float = 0.0001
MIN_RANGE_MUTATION: float = -5.0
MAX_RANGE_MUTATION: float = 5.0


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


def model_crossover(parent1_weight: list, parent2_weight: list):
    """
    This function make a crossover of tow models
    :param parent1_weight: the weights of the firs model
    :param parent2_weight:the weights of the second model
    :return: new weight from a crossover of the two parents
    """
    new_weight: list = []

    # get the shape of the wight
    shapes: [tuple] = [a.shape for a in parent1_weight]

    # flatten weight
    genes1: np.array = np.concatenate([a.flatten() for a in parent1_weight])
    genes2: np.array = np.concatenate([a.flatten() for a in parent2_weight])

    # create the split coordinate
    split = random.randint(0, len(genes1) - 1)

    # make the crossover from the two parents
    child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())

    # give the good shape to the weight of the child
    index = 0
    for shape in shapes:
        size = np.product(shape)
        new_weight.append(child1_genes[index: index + size].reshape(shape))
        index += size

    return new_weight


def model_mutation(model_weight: list,
                   mutation_rate: float =
                   MUTATION_RATE,
                   min_range_mutation: float = MIN_RANGE_MUTATION,
                   max_range_mutation: float = MAX_RANGE_MUTATION):
    """
    This function add some mutation in the model weight
    :param model_weight: model weight where mutation will be added
    :param mutation_rate: 1 = 100% the probability of a mutation
    :param min_range_mutation the minimum range of a random mutation
    :param max_range_mutation the maximum range of a random mutation
    :return: the model with mutation
    """

    # get the shape of the wight
    shapes: [tuple] = [a.shape for a in model_weight]

    # flatten weight
    genes: np.array = np.concatenate([a.flatten() for a in model_weight])

    # create mutation
    for i in range(len(genes)):
        if random.uniform(0, 1) < mutation_rate:
            genes[i] = random.uniform(min_range_mutation, max_range_mutation)

    new_weight: list = []

    # give the good shape to the muted weight
    index = 0
    for shape in shapes:
        size = np.product(shape)
        new_weight.append(genes[index: index + size].reshape(shape))

    return new_weight



if __name__ == '__main__':
    m = Model2048()
    m.summary()

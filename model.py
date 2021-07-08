"""
This script contains all the functions related to the model
"""

import tensorflow as tf
import numpy as np
import random
from math import ceil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Dropout
from game import MOVES_POSSIBLE, ALL_BLOCK_POSSIBLE, GRID_SIZE_X, GRID_SIZE_Y
from tensorflow.keras.utils import to_categorical
from copy import deepcopy


EPS: float = 0.4  # probability of playing a random move
# list of all actions possible for the model
LIST_ACTIONS: [int] = [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]

MUTATION_RATE: float = 0.0001
MIN_RANGE_MUTATION: float = -5.0
MAX_RANGE_MUTATION: float = 5.0

# parent is for the rate of parent in the new generation
# children is for the rate of children in the new generation
# child must have a inferior or equal to parent
# new is the rate of new random model in the new generation
# the sum of child, parent and new must be equal to 1
GENERATION_PRESET: dict = {"parent": 0.4, "children": 0.4, "new": 0.2}  # this preset is a model for a new generation


# model

class Model2048(Sequential):
    """
    Create the main model for 2048
    """

    def __init__(self):
        super().__init__()

        """
        # create the model
        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE * MOVES_POSSIBLE,
                   (1, 2),
                   padding="same",
                   input_shape=(GRID_SIZE_Y, GRID_SIZE_X, ALL_BLOCK_POSSIBLE),
                   )
        )
        self.add(LeakyReLU())

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE * MOVES_POSSIBLE,
                   (2, 1),
                   padding="same",
                   )
        )
        self.add(LeakyReLU())

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE,  # * MOVES_POSSIBLE
                   (1, 1),
                   padding="same",
                   )
        )
        self.add(LeakyReLU())

        self.add(Flatten())

        self.add(Dropout(0.2))

        self.add(Dense(256))
        self.add(LeakyReLU())

        self.add(Dropout(0.2))

        self.add(Dense(4, activation="softmax"))

        self.compile(optimizer="adam", loss="huber_loss")

        """

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE * MOVES_POSSIBLE,
                   (1, 2),
                   padding="same",
                   input_shape=(GRID_SIZE_Y, GRID_SIZE_X, ALL_BLOCK_POSSIBLE),
                   )
        )
        self.add(LeakyReLU())

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE * MOVES_POSSIBLE,
                   (2, 1),
                   padding="same",
                   )
        )
        self.add(LeakyReLU())

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE,  # * MOVES_POSSIBLE
                   (1, 1),
                   padding="same",
                   )
        )
        self.add(LeakyReLU())

        self.add(Flatten())

        self.add(Dropout(0.2))

        self.add(Dense(256))
        self.add(LeakyReLU())

        self.add(Dropout(0.2))

        self.add(Dense(4, activation="softmax"))

        self.compile(optimizer="RMSprop", loss="huber_loss")

        """
        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE * MOVES_POSSIBLE,
                   (1, 2),
                   padding="same",
                   input_shape=(GRID_SIZE_Y, GRID_SIZE_X, 1),
                   )
        )
        self.add(LeakyReLU())

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE * MOVES_POSSIBLE,
                   (2, 1),
                   padding="same",
                   )
        )
        self.add(LeakyReLU())

        self.add(
            Conv2D(ALL_BLOCK_POSSIBLE,  # * MOVES_POSSIBLE
                   (1, 1),
                   padding="same",
                   )
        )
        self.add(LeakyReLU())

        self.add(Flatten())

        self.add(Dropout(0.2))

        self.add(Dense(256))
        self.add(LeakyReLU())

        self.add(Dropout(0.2))

        self.add(Dense(4, activation="softmax"))

        self.compile(optimizer="adam", loss="huber_loss")
        """
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

    # @tf.function
    def model_action(self, grid):
        """
        This function return the input of the model
        :param grid: a 2048 grid
        :return: input of the model
        """
        return self(np.array([grid_to_input(grid)], dtype=np.float32), training=False).numpy()[0]

    def take_action(self, grid, eps: float = EPS):
        """
        This function will take sometime a random action and sometime a model action
        :param grid: a sequence of indicators of length SEQUENCE_LENGTH
        :param eps: probability of playing a random move
        :return: the action take
        """
        if random.random() < eps:
            # take random action
            action = deepcopy(random.choice(LIST_ACTIONS))

        else:
            # let model choose a action
            action = self.model_action(grid)
            """
            returned_list = [0, 0, 0, 0]
            returned_list[np.argmax(action)] = 1

            return returned_list"""

        return action


def normalization(x, min_range, max_range):
    """
    Normalization function
    :param x: List of value to normalize
    :param min_range: Minimum range for norm
    :param max_range: Maximum range for norm
    :return: array normalize
    """

    x_max = max(x.flatten().tolist())
    x_min = min(x.flatten().tolist())

    norm = min_range + ((x - x_min) * (max_range - min_range)) / (x_max - x_min)

    return norm


def grid_to_input(grid):
    """
    This function transform the grid to a model input
    :param grid: a 2048 grid
    :return: the input for the model
    """


    # MULTI LAYER PERCEPTION
    # transform to categorical
    grid = to_categorical(np.log2(grid + 1) - 1, 18).tolist()

    # remove 0
    for y in range(4):
        for x in range(4):
            del grid[y][x][-1]

    return np.array(grid)

"""
    # ONE LAYER PERCEPTION
    grid = grid * 2
    grid[grid == 0] = 2
    grid = np.log2(grid)
    grid -= 1
    grid = normalization(grid, 0, 1)
    grid = np.reshape(grid, grid.shape + (1, ))

    return grid
"""

# genetic algorithm

def model_crossover(parent1_weight: list, parent2_weight: list) -> list:
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
                   mutation_rate: float = MUTATION_RATE,
                   min_range_mutation: float = MIN_RANGE_MUTATION,
                   max_range_mutation: float = MAX_RANGE_MUTATION) -> list:
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


def new_generation(all_gen_weight: list,
                   all_gen_score: [int],
                   generation_preset: dict = None) -> list:
    """
    this function return a new generation from a older generation
    :param all_gen_weight: a list that contain all model's weight (should be a list of list of array)
                           you must get weight of all models
    :param all_gen_score: a list that contain the score of each model (should be a list of int)
    warning : index of all_gen_weight must correspond with index of all_gen_score
    :param generation_preset: the presset for generation
    :return: a new generation from a older generation
    """
    # set generation to default if parameter if None
    if generation_preset is None:
        generation_preset = GENERATION_PRESET

    # sort the score from the biggest to the smalest
    best_all_gen_score = sorted(all_gen_score, reverse=True)

    # create a list that store best model
    best_models: list = []

    # select best model
    for i in range(ceil(len(all_gen_weight) * generation_preset["parent"])):
        # get the index of the maximum score in the list
        index_best: int = all_gen_score.index(best_all_gen_score[i])

        # add the best model to the list of best model
        best_models.append(all_gen_weight[index_best])

    # create children
    children_models: list = []

    for i in range(ceil(len(all_gen_weight) * generation_preset["children"])):
        children_models.append(
            model_crossover(best_models[i], best_models[i - 1])
        )

    # create mutation
    parent_children_list: list = best_models + children_models

    for i in range(len(parent_children_list)):
        parent_children_list[i] = model_mutation(parent_children_list[i])

    # add random model
    random_models: list = []

    for i in range(ceil(len(all_gen_weight) * generation_preset["new"])):
        _temp_m = Model2048()
        _temp_w = _temp_m.get_weights()
        random_models.append(_temp_w)

    # create the full new gen
    new_gen: list = parent_children_list + random_models

    return new_gen


if __name__ == '__main__':
    m = Model2048()
    m.summary()

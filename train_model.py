"""
This function will train the model
"""

# import multiprocessing
# from multiprocessing import Manager
import threading
import tensorflow as tf
import numpy as np
import time
import random
from copy import deepcopy
import os

import matplotlib.pyplot as plt
import matplotlib
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

from game import Game2048
from game import CASE_COLOR_GAME, ALL_BLOCK_POSSIBLE, GRID_SIZE_Y, GRID_SIZE_X
from model import Model2048
from model import new_generation, grid_to_input, model_mutation

matplotlib.use('TkAgg')

GUI: bool = True  # if you want see your models train
Q_LEARNING_DURING_ENV: bool = True  # enable training during env simulation
Q_LEARNING_AFTER: bool = True
GENETIC_ALGORITHM: bool = True

# EPOCHS: int = 10_000
EPOCHS: int = 10_000
HISTORY_SIZE: int = 100_000

# create gui disposition
Y_MODEL: int = 2
X_MODEL: int = 3
TOTAL_MODEL: int = int(Y_MODEL * X_MODEL)

# color map for gui
FULL_MAP: list = [CASE_COLOR_GAME[0]] + [CASE_COLOR_GAME[2 ** i] for i in range(1, ALL_BLOCK_POSSIBLE + 1)]
FULL_BOUNDS: list = [-0.5] + [2 ** i - 0.5 for i in range(1, ALL_BLOCK_POSSIBLE + 2)]
COLOR_MAP = matplotlib.colors.ListedColormap(FULL_MAP)
NORM = matplotlib.colors.BoundaryNorm(FULL_BOUNDS, COLOR_MAP.N)

# q learning
LEARNING_RATE: float = 0.1  # alpha
DISCOUNT_FACTOR: float = 0.99  # gamma

# model gradiant tape
MODEL_OPTIMIZER = tf.keras.optimizers.Adam()
MODEL_LOSS = tf.keras.losses.Huber()


# @tf.function
def environment(index) -> None:
    """
    This function will create a environment for each model
    :param index: the index of the environment
    :return: None
    """

    # while the game is not finished
    while not list_game[index].check_end(list_game[index].grid):
        t = time.time()
        # add the actual grid to the history
        board_history_x[index].append(list_game[i].grid)

        # get the action of the model for the actual grid
        model_action = list_model[index].take_action(list_game[index].grid)

        # categorical to value
        model_action_index = np.argmax(model_action)

        # add to history the action choose by the model
        board_history_y[index].append(model_action)

        # simulate the action
        new_grid, reward = list_game[index].action(list_game[index].grid, model_action_index)

        # add reward to history
        board_history_r[index].append(reward)

        # do simple q learning for only this model
        # get model action for future
        future_action = list_model[index].take_action(new_grid)
        future_action = future_action[np.argmax(future_action)]  # = Q(st+1, at+1)

        # Q(st, at) = Q(st, at) + α *(rt + Ɣ * Q(st+1, at+1) - Q(st, at))

        model_action[model_action_index] = model_action[model_action_index] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * future_action - model_action[model_action_index])

        # train model
        """
        list_model[index].fit(
            np.array([grid_to_input(list_game[index].grid)]),
            np.array([model_action]),
            verbose=0,
            batch_size=1)"""
        # train model
        if Q_LEARNING_DURING_ENV:
            with tf.GradientTape() as tape:
                logits = list_model[index](
                    np.array([grid_to_input(list_game[index].grid)], dtype=np.float32), training=True
                )

                # Compute the loss value for this minibatch.
                loss_value = MODEL_LOSS(np.array([model_action]), logits)

            grads = tape.gradient(loss_value, list_model[index].trainable_weights)

            MODEL_OPTIMIZER.apply_gradients(zip(grads, list_model[index].trainable_weights))

        # set new environement
        list_game[index].grid = new_grid
        list_game[index].score += reward

    list_score[index] = list_game[index].score
    list_max_block[index] = max(list_game[index].grid.flatten().tolist())


def gui() -> None:
    """
    This function  create a gui that show the advancement of all games
    :return: None
    """
    # create gui
    fig, axs = plt.subplots(Y_MODEL, X_MODEL)

    for y in range(Y_MODEL):
        for x in range(X_MODEL):
            # creation de l'affichage de chaque terrain
            temp_mat = axs[y, x].matshow(
                np.array([[0 for _ in range(GRID_SIZE_X)] for _ in range(GRID_SIZE_Y)]),
                cmap=COLOR_MAP,
                norm=NORM)
            temp_mat.axes.xaxis.set_visible(False)
            temp_mat.axes.yaxis.set_visible(False)

            graph_list.append(temp_mat)

    # create a tkinter window that contain all matplotlib game graph
    window = Tk()
    window.config(bg="white")
    window.title("2048 AI")

    canvas = FigureCanvasTkAgg(fig, window)
    canvas.get_tk_widget().pack(side="top", fill='both', expand=True)

    ani = animation.FuncAnimation(fig,
                                  update_graph,
                                  interval=1)

    window.mainloop()


def update_graph(i) -> None:
    """
    This function update the graph for each game
    :param i:
    :return: None
    """
    for z in range(TOTAL_MODEL):
        # afficher sur le graph
        try:
            graph_list[z].set_data(list_game[z].grid)
        except ValueError as _:
            pass


if __name__ == '__main__':
    # list for animated graph
    graph_list = []

    # create models
    list_model: list = [Model2048() for _ in range(TOTAL_MODEL)]

    # create game for each model
    list_game: list = [Game2048() for _ in range(TOTAL_MODEL)]

    # start window
    if GUI:
        thread_window = threading.Thread(target=gui)
        thread_window.start()

    # for each epochs
    for epoch in range(EPOCHS):

        t0 = time.time()

        action_taken: int = 0

        # reset games
        list_game: list = [Game2048() for _ in range(TOTAL_MODEL)]

        # for scores
        list_score: [int] = [0 for _ in range(TOTAL_MODEL)]
        list_max_block: [int] = [0 for _ in range(TOTAL_MODEL)]

        # create a board history
        board_history_x: list = [[] for _ in range(TOTAL_MODEL)]
        # create a move history
        board_history_y: list = [[] for _ in range(TOTAL_MODEL)]
        # create reward history
        board_history_r: list = [[] for _ in range(TOTAL_MODEL)]

        # create score for each env
        score: list = [0 for _ in range(TOTAL_MODEL)]

        multi_env: list = []

        # time for games
        t1 = time.time()

        # start every thread for each env
        for i in range(TOTAL_MODEL):
            multi_env.append(
                threading.Thread(target=environment, args=(i,))
            )
            multi_env[-1].start()

        # join
        for env in multi_env:
            env.join()

        t1 = time.time() - t1

        t2 = time.time()

        # genetic algorithm
        if GENETIC_ALGORITHM:

            # get weight of all model
            model_weight = [m.get_weights() for m in list_model]

            new_gen = new_generation(model_weight, list_score)

            # add mutation
            for i in range(TOTAL_MODEL):
                new_gen[i] = model_mutation(new_gen[i])

            # load models
            for i, model in enumerate(list_model):
                model.set_weights(new_gen[i])

        t2 = time.time() - t2

        # print stats
        print(f"----- {epoch + 1} -----")
        print(f"Scores :")
        print(f"\t- Average:           {round(sum(list_score) / len(list_score), 1)}")
        print(f"\t- Maximum:           {max(list_score)}")
        print(f"\t- Minimum:           {min(list_score)}")
        print(f"\t- Median block:      {sorted(list_max_block)[TOTAL_MODEL // 2]}")
        print(f"\t- Maximum block:     {max(list_max_block)}")
        print(f"Time :")
        print(f"\t- Total:             {round(time.time() - t0, 1)}")
        print(f"\t- Games simulation:  {round(t1, 1)}")
        print(f"\t- Genetic algorithm: {round(t2, 1)}")



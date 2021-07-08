"""
This script will train a q model
"""

import tensorflow as tf
import numpy as np
import time
import threading

import matplotlib.pyplot as plt
import matplotlib
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

from game import Game2048
from game import CASE_COLOR_GAME, ALL_BLOCK_POSSIBLE, GRID_SIZE_Y, GRID_SIZE_X
from model import Model2048, grid_to_input

matplotlib.use('TkAgg')

GUI: bool = True  # if you want see your models train

EPOCHS: int = 100_000
HISTORY_SIZE: int = 10_000

# color map for gui
FULL_MAP: list = [CASE_COLOR_GAME[0]] + [CASE_COLOR_GAME[2 ** i] for i in range(1, ALL_BLOCK_POSSIBLE + 1)]
FULL_BOUNDS: list = [-0.5] + [2 ** i - 0.5 for i in range(1, ALL_BLOCK_POSSIBLE + 2)]
COLOR_MAP = matplotlib.colors.ListedColormap(FULL_MAP)
NORM = matplotlib.colors.BoundaryNorm(FULL_BOUNDS, COLOR_MAP.N)

# q learning
LEARNING_RATE: float = 1  # alpha
DISCOUNT_FACTOR: float = 0.99  # gamma

# model gradiant tape
MODEL_OPTIMIZER = tf.keras.optimizers.SGD()
MODEL_LOSS = tf.keras.losses.MeanSquaredError()


def gui() -> None:
    """
    This function  create a gui that show the advancement of all games
    :return: None
    """
    global mat, axs
    # create gui
    fig, axs = plt.subplots(2, 2)

    # the main game
    mat = axs[0, 0].matshow(
        np.array([[0 for _ in range(GRID_SIZE_X)] for _ in range(GRID_SIZE_Y)]),
        cmap=COLOR_MAP,
        norm=NORM)
    mat.axes.xaxis.set_visible(False)
    mat.axes.yaxis.set_visible(False)
    axs[0, 0].set_title("2048 AI Game")

    # init score per game graph
    axs[0, 1].set_title("Score per game")
    axs[0, 1].plot([0], [0], color="blue")

    # init score per game
    axs[1, 1].set_title("Steps per game")
    axs[1, 1].plot([0], [0], color="blue")

    # init max block per game
    axs[1, 0].set_title("Max block per game")
    axs[1, 1].plot([0], [0], color="blue")

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


def update_graph(i):
    try:
        mat.set_data(game.grid)

        axs[0, 1].clear()
        axs[0, 1].set_title("Score per game")
        axs[0, 1].plot([i for i in range(len(score_per_game_history))], score_per_game_history)

        axs[1, 1].clear()
        axs[1, 1].set_title("Steps per game")
        axs[1, 1].plot([i for i in range(len(steps_per_game_history))], steps_per_game_history)

        axs[1, 0].clear()
        axs[1, 0].set_title("Max block per game")
        axs[1, 0].plot([i for i in range(len(max_block_per_game_history))],
                       np.log2(np.array(max_block_per_game_history) + 0.1))
    except ValueError as _:
        pass


if __name__ == '__main__':

    # init a game for graph
    game = Game2048()

    # init variables for the graph
    mat = None  # the main grid
    axs = None

    # init list that contain all values
    score_per_game_history: list = [0]
    steps_per_game_history: list = [0]
    max_block_per_game_history: list = [0]

    # gui
    if GUI:
        thread_window = threading.Thread(target=gui)
        thread_window.start()

    # create the model
    model = Model2048()
    # create reqaard model
    reward_model = Model2048()

    # to train the model
    X_TRAIN: list = []
    Y_TRAIN: list = []

    for epoch in range(EPOCHS):
        print(epoch)
        # init a game
        game = Game2048()

        # number of action realized
        n_action: int = 0

        while not game.check_end(game.grid):

            steps_per_game_history[epoch] = n_action

            # get the action of the model for the actual grid
            model_action = model.take_action(game.grid, eps=0.1)
            # print(model_action)
            # categorical to value
            model_action_index: int = int(np.argmax(model_action))
            # simulate the action
            new_grid, reward = game.action(game.grid, model_action_index)

            # q learning
            # get model action for future
            future_action = np.array(reward_model.model_action(new_grid))
            # future_action = future_action[np.argmax(future_action)]  # = Q(st+1, at+1)

            # Q(st, at) = Q(st, at) + α *(rt + Ɣ * Q(st+1, at+1) - Q(st, at)) :
            model_action[model_action_index] = model_action[model_action_index] + LEARNING_RATE * (
                    (reward - 1) + DISCOUNT_FACTOR * max(future_action) - model_action[model_action_index])
            # model_action = model_action + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max(future_action) - model_action)

            """with tf.GradientTape() as tape:
                logits = model(
                    np.array([grid_to_input(game.grid)], dtype=np.float32), training=True
                )

                # Compute the loss value for this minibatch.
                loss_value = MODEL_LOSS(np.array([model_action]), logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            MODEL_OPTIMIZER.apply_gradients(zip(grads, model.trainable_weights))
            """
            if new_grid.tolist() != game.grid.tolist():
                n_action += 1
                Y_TRAIN.append(model_action)
                X_TRAIN.append(grid_to_input(new_grid))

            # update game
            game.grid = new_grid
            game.score += reward

            score_per_game_history[epoch] += reward

            # max block
            max_block = max(game.grid.flatten().tolist())
            max_block_per_game_history[epoch] = max_block

        score_per_game_history.append(0)
        steps_per_game_history.append(0)
        max_block_per_game_history.append(0)

        # train the model
        model.fit(np.array(X_TRAIN), np.array(Y_TRAIN), verbose=0, epochs=1)

        # update reward model
        reward_model.set_weights(model.get_weights())

        if len(X_TRAIN) > HISTORY_SIZE:
            X_TRAIN: list = []
            Y_TRAIN: list = []

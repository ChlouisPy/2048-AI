"""
Play 2048 (without AI for the moment)
"""
# wor web option
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys

import numpy as np
import time
import xxhash
from copy import deepcopy

import game

# all moves possibles in the game
MOVES_DRIVER: dict = {3: Keys.UP, 0: Keys.LEFT, 1: Keys.DOWN, 2: Keys.RIGHT}

LIM_EXPLORE: int = 4  # move ahead explored


# website interaction function
def get_game_grid() -> np.array:
    """
    This function will get on the webpage the grid of 2048 and return a numpy array of the grid
    :return: np.array of the grid
    """
    # the grid that will be returned
    grid: np.array = np.zeros((4, 4))

    # get the content of the grid
    grid_content = DRIVER.find_element_by_class_name("tile-container")
    soup = BeautifulSoup(grid_content.get_attribute('innerHTML'), features="lxml")
    all_block = soup.findAll("div")
    all_block = [all_block[i] for i in range(0, len(all_block), 2)]

    for e in all_block:
        y: int = int(str(e).split(" tile-position-")[1][0]) - 1
        x: int = int(str(e).split(" tile-position-")[1][2]) - 1
        value: int = int(str(e).split("tile")[2]) * -1
        grid[y][x] = value

    return grid


def play_move(key_index: int) -> None:
    """
    This function will play a move in the game
    :param key_index: the key wanted by the program
    :return: None
    """
    DRIVER.find_element_by_css_selector("body").send_keys(MOVES_DRIVER[key_index])


# game functions
def action(grid: np.array, axis: int = 0) -> np.array:
    """
    This function will play one on the 4 actions possibles in 2048 (↑, ←, ↓, →)
    :param grid: the 2048 grid where to play
    :param axis: the number of rotation to do before playing 0: ↑ 1: → 2: ↓ 3: ←
    :return: the new grid
    """
    # rotate the grid to place it in the right direction
    grid = deepcopy(np.rot90(grid, axis))

    # 1. pack
    grid = pack(grid)
    # 2. addition
    grid, _score = addition(grid)

    # 3. pack
    grid = pack(grid)

    # replace the grid ine the right direction
    grid = np.rot90(grid, 4 - axis)

    # return the grid with a new action
    return grid, _score


def pack(grid: np.array) -> np.array:
    """
    this function will pack the 2048 grid on the top of the grid
    :param grid: the 2048 grid to pack
    :return: the packed grid
    """

    # for every columns
    for x in range(4):
        # from the top to the bottom
        for y in range(1, 4):

            # check if the case x, y in the grid is a block, if it is a block it can move else it can't
            if grid[y][x] != 0:
                # now try to pack on the top of the grid the block
                for i in range(1, y + 1):

                    # if it can't pack anymore break
                    if grid[y - i][x] != 0:
                        break
                    # if it can pack pack one step on the top
                    else:
                        grid[y - i][x] = int(grid[y - i + 1][x])
                        grid[y - i + 1][x] = 0

    return grid


def addition(grid: np.array) -> tuple:
    """
    This function make all additions in the grid
    :param grid: the 2048 grid
    :return: the 2048 grid with additions and the score realized
    """

    score: int = 0

    # for every columns
    for x in range(4):

        # a + b or c + d
        if grid[0][x] == grid[1][x] or grid[2][x] == grid[3][x]:
            # c + d
            if grid[2][x] == grid[3][x]:
                add: int = grid[2][x] + grid[3][x]
                # set the news values in the grid
                grid[2][x], grid[3][x] = add, 0
                # add the value of the two blocks in the score
                score += add

            # a + b
            if grid[0][x] == grid[1][x]:
                add: int = grid[0][x] + grid[1][x]
                # set the news values in the grid
                grid[0][x], grid[1][x] = add, 0
                # add the value of the two blocks in the score
                score += add

        else:
            # b + c
            if grid[1][x] == grid[2][x]:
                add: int = grid[1][x] + grid[2][x]
                # set the news values in the grid
                grid[1][x], grid[2][x] = add, 0
                # add the value of the two blocks in the score
                score += add

    return grid, score


def check_end(grid: np.array) -> bool:
    """
    This function return if the game of 2048 is finished
    :param grid: the 2048 grid
    :return: True if the game is finished or False if the game is not finished
    """
    # check every possibles addition

    # check if there is 0 in the grid
    if 0 in grid.flatten():
        return False

    # column checking
    for x in range(4):
        for y in range(1, 4):
            # if two block can be added in column
            if grid[y][x] == grid[y - 1][x]:
                return False

    # row checking
    for y in range(4):
        for x in range(1, 4):
            # if two block can be added in row
            if grid[y][x] == grid[y][x - 1]:
                return False

    return True


# explore function
def hash_array(array: np.array) -> str:
    """
    This function hash a numpy array
    :param array: a numpy array
    :return: hashed numpy array
    """
    return xxhash.xxh128(np.array(array)).hexdigest()


def explore_node(grid: np.array, main_index: int, layer: int) -> tuple:
    """
    This function will explore a derivation
    :param grid: a 2048 grid to explore
    :param main_index: the index of the base move
    :param layer: layer of exploration
    :return: move generated, reward
    """
    # check if the limit of layer exploration is exceeded
    if layer >= LIM_EXPLORE:
        return 0, 0

    # check if the game is already ended
    if check_end(grid):
        return 0, sum(grid.flatten().tolist()) * -1

    total_score_returned: int = 0
    total_moves_returned: int = 0

    # start exploring each moves
    for j in range(4):

        # create a derivation
        f_new_grid, f_score = action(grid, j)
        f_new_grid_hash = hash_array(f_new_grid)

        total_moves_returned += 1

        # check if the board is in the hash table
        if f_new_grid_hash in hash_table:
            total_score_returned += hash_table[f_new_grid_hash]
            total_moves_returned += hash_table_move[f_new_grid_hash]

        # check if the move is possible
        elif not (f_new_grid == grid).all():
            # add the score in the sum
            total_score_returned += f_score

            score_per_move: int = 0
            moves_per_move: int = 0

            # spawn all randoms blocks possibles
            for f_block in [2, 4]:
                for f_y in range(4):
                    for f_x in range(4):
                        # check if a block can spawn here
                        if new_grid[f_y][f_x] == 0:
                            f_new_grid_exploration = deepcopy(f_new_grid)
                            f_new_grid_exploration[f_y][f_x] = f_block

                            _nm, _rw = explore_node(f_new_grid_exploration, main_index, layer + 1)

                            score_per_move += _rw
                            moves_per_move += _nm

                            # add the new board in the hash table
                            hash_table[hash_array(f_new_grid_exploration)] = _rw
                            hash_table_move[hash_array(f_new_grid_exploration)] = _nm

            # calculate the total reward
            total_score_returned += f_score + score_per_move
            total_moves_returned += moves_per_move

            hash_table[f_new_grid_hash] = f_score + score_per_move
            hash_table_move[f_new_grid_hash] = moves_per_move

    return total_moves_returned, total_score_returned


if __name__ == '__main__':
    # create the system to play online
    DRIVER = webdriver.Firefox()

    # open 2048 web site
    DRIVER.get("https://play2048.co/")

    GAME_RUN: bool = True

    STEP: int = 0

    while GAME_RUN:
        t = time.time()

        # update number os step
        STEP += 1

        # reset hash table
        hash_table: dict = {}
        hash_table_move: dict = {}

        # get grid
        actual_grid: np.array = get_game_grid()

        # reset reward expected
        reward_final: list = [0, 0, 0, 0]
        n_moves_final: list = [0, 0, 0, 0]

        possible: list = [True, True, True, True]

        # start exploration
        for i in range(4):

            # create a derivation
            new_grid, score = action(actual_grid, i)

            # check if the derivation is possible
            if not (new_grid == actual_grid).all():
                # spawn random numbers

                for block in [2, 4]:
                    for y in range(4):
                        for x in range(4):
                            # check if a block can spawn here
                            if new_grid[y][x] == 0:
                                new_grid_exploration = deepcopy(new_grid)
                                new_grid_exploration[y][x] = block

                                nm, rw = explore_node(new_grid_exploration, i, 1)

                                reward_final[i] += rw
                                n_moves_final[i] += nm
            else:
                possible[i] = False

        # calculate the best reward
        best_reward = [reward_final[i] / (n_moves_final[i] + 0.00001) for i in range(4)]

        for i in range(4):
            if not possible[i]:
                best_reward[i] = -1_000_000_000_000

        best_move = best_reward.index(max(best_reward))

        play_move(best_move)

        # print stats
        print(f"---------- {STEP} ----------")
        print(f"{sum(n_moves_final)} moves calculated in {round(time.time() - t, 1)}")
        print(f"Max reward: {max(reward_final)} | Min reward: {min(reward_final)}")
        print(f"Radio max: {round(max(best_reward), 1)} | Radio min: {round(min(best_reward), 1)}")
        print(best_reward)
        print(n_moves_final)

    DRIVER.close()

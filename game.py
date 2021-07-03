"""
This script contain all the functions related to the 2048 game
"""

import random
from copy import deepcopy
import numpy as np

# the probability of spawning 4
RATE_FOUR: float = 0.1

# board size
GRID_SIZE_X: int = 4
GRID_SIZE_Y: int = 4


class Game2048:
    def __init__(self):
        # create the 2048 grid with a size of 4 by 4
        self.grid: np.array = np.array([[0 for x in range(GRID_SIZE_X)] for y in range(GRID_SIZE_Y)])
        # score of the game
        self.score = 0

        # spawn the two first number
        for _ in range(2):
            self.grid = self.spawn_number(self.grid)

    @staticmethod
    def spawn_number(grid: np.array) -> np.array:
        """
        this function spawn 1 random number between 2 and 4 in a random place in the 2048 grid
        :param grid: the grid where to place the new random bloc
        :return: the grid with one more bloc
        """
        # create two random coordinate for the place where the new block will appear
        x: int = random.randint(0, GRID_SIZE_X - 1)
        y: int = random.randint(0, GRID_SIZE_Y - 1)

        # check if it is possible to spawn the block at this place
        while grid[y][x] != 0:
            # if it can spawn the block create a new random coordinate
            x: int = random.randint(0, GRID_SIZE_X - 1)
            y: int = random.randint(0, GRID_SIZE_Y - 1)

        # if the coordinate is free place the block
        if random.uniform(0, 1) < RATE_FOUR:
            # place a 4
            grid[y][x] = 4
        else:
            # place a 2
            grid[y][x] = 2

        # return the grid with a new block
        return grid

    def action(self, grid: np.array, axis: int = 0) -> np.array:
        """
        This function will play one on the 4 actions possibles in 2048 (↑, ←, ↓, →)

        # how it works :
        1. it packs the grid
        2. it makes addition
        3. it packs a second time the grid
        4. place a new random block

        :param grid: the 2048 grid where to play
        :param axis: the number of rotation to do before playing 0: ↑ 1: → 2: ↓ 3: ←
        :return: the new grid
        """
        # rotate the grid to place it in the right direction
        grid = np.rot90(grid, axis)

        grid_copy = deepcopy(grid)

        # 1. pack
        grid = self.pack(grid)
        # 2. addition
        grid, score = self.addition(grid)
        self.score += score
        # 3. pack
        grid = self.pack(grid)
        # 4.place a new block
        if grid.tolist() != grid_copy.tolist():
            # place a new block only if something move in the grid
            grid = self.spawn_number(grid)

        # replace the grid ine the right direction
        grid = np.rot90(grid, 4 - axis)

        # return the grid with a new action
        return grid

    @staticmethod
    def pack(grid: np.array) -> np.array:
        """
        this function will pack the 2048 grid on the top of the grid
        :param grid: the 2048 grid to pack
        :return: the packed grid
        """

        # for every columns
        for x in range(GRID_SIZE_X):
            # from the top to the bottom
            for y in range(1, GRID_SIZE_Y):

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

    @staticmethod
    def addition(grid: np.array) -> tuple:
        """
        This function make all additions in the grid

        # there are 2 steps in the addition function :

        this is a column in a grid of 2048 :
        [a,
         b,
         c,
         d]

         first the function calculates the two sides like that :
         [a, |____ a + b
         b,  |
         c,  |____ c + d
         d]  |

         and in a second time, if the first step do nothing, the function calculates the center
         [a,
          b,  |____ b + c
          c,  |
          d]

        :param grid: the 2048 grid
        :return: the 2048 grid with additions and the score realized
        """

        score: int = 0

        # for every columns
        for x in range(GRID_SIZE_X):
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


if __name__ == '__main__':
    G = Game2048()
    while True:
        # os.system("cls")

        print(G.grid)
        m = int(input("move >>> "))
        if m == 8:
            G.action(G.grid, 0)
        elif m == 6:
            G.action(G.grid, 1)
        elif m == 4:
            G.action(G.grid, 3)
        elif m == 2:
            G.action(G.grid, 2)

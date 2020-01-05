import math

import numpy as np

from Player import Player


class SkillshotGame(object):

    def __init__(self):
        self.board_size = (250, 250)
        self.board = np.zeros(self.board_size, dtype=int)

        self.player1 = Player([150, 100], self.board_size)
        self.player2 = Player([150, 200], self.board_size)

    def get_board(self):
        board = self.board.copy()
        # append players
        for player, player_colour_index, pointer_colour_index in zip((self.player1, self.player2), [1, 2], [3, 4]):
            for index_y, row in enumerate(player.shape_image):
                for index_x, item in enumerate(row):
                    if item is not 0:
                        board[index_x + player.pos[0], index_y + player.pos[1]] = player_colour_index
                    # append direction indicator
                    if (index_x == math.floor(-math.sin(player.rotation) * player.shape_size[0] / 2 + player.shape_size[0] / 2) and
                        index_y == math.floor(-math.cos(player.rotation) * player.shape_size[1] / 2 + player.shape_size[1] / 2)):
                        board[index_x + player.pos[0], index_y + player.pos[1]] = pointer_colour_index

        # append projectile 1
        # append projectile 2

        return board

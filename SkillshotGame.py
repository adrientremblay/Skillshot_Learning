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
        # append player 1
        for index_y, row in enumerate(self.player1.shape):
            for index_x, item in enumerate(row):
                if item is not 0:
                    board[index_x + self.player1.pos[0], index_y + self.player1.pos[1]] = 1

        # append player 2
        for index_y, row in enumerate(self.player2.shape):
            for index_x, item in enumerate(row):
                if item is not 0:
                    board[index_x + self.player2.pos[0], index_y + self.player2.pos[1]] = 2

        # append projectile 1
        # append projectile 2

        return board

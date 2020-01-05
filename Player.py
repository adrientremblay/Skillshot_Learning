import math

import numpy as np


class Player(object):
    shape = [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1]]
    speed_move = 5
    speed_look = 0.1

    def __init__(self, pos, board_dim):
        self.pos = pos
        self.rotation = 0
        self.board_dim = board_dim

    def move_look_left(self):
        self.rotation += self.speed_look
        print(self.rotation)

    def move_look_right(self):
        self.rotation -= self.speed_look

    def move_forwards(self):
        new_pos_x = int(round(self.pos[0] - math.sin(self.rotation) * self.speed_move))
        new_pos_y = int(round(self.pos[1] - math.cos(self.rotation) * self.speed_move))

        if self.check_pos_valid(new_pos_x, new_pos_y):
            self.pos[0] = new_pos_x
            self.pos[1] = new_pos_y

    def move_backwards(self):
        new_pos_x = int(round(self.pos[0] + math.sin(self.rotation) * self.speed_move))
        new_pos_y = int(round(self.pos[1] + math.cos(self.rotation) * self.speed_move))

        if self.check_pos_valid(new_pos_x, new_pos_y):
            self.pos[0] = new_pos_x
            self.pos[1] = new_pos_y

    def check_pos_valid(self, check_x, check_y):
        if check_x + len(self.shape[0]) <= 250 and check_x >= 0 and check_y + len(self.shape) <= 250 and check_y >= 0:
            return True
        else:
            return False

    def move_shoot_projectile(self):
        pass

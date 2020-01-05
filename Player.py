import math

import numpy as np

from Projectile import Projectile


class Player(object):
    shape_image = [[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]]
    speed_move = 3
    speed_look = 0.1
    speed_projectile = 5

    def __init__(self, pos, board_dim):
        self.pos = pos
        self.rotation = 0
        self.board_dim = board_dim
        self.shape_size = (len(self.shape_image[0]), len(self.shape_image))

        self.projectile = Projectile(self.speed_projectile, (0, 0), board_dim)

    def move_look_left(self):
        self.rotation += self.speed_look

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
        # checks if a position if within the board bounds
        if check_x + self.shape_size[0] <= 250 and check_x >= 0 and check_y + self.shape_size[1] <= 250 and check_y >= 0:
            return True
        else:
            return False

    def move_shoot_projectile(self):
        # sets projetile position to  player position
        self.projectile.set_position(self.pos)
        # sets projectile direction to rotation (in rad, convert to gradent)
        self.projectile.set_direction(math.tan(self.rotation))
        # sets projectile to valid
        self.projectile.valid = True
        print(self.projectile.direction)

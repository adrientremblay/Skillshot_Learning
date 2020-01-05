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
    speed_look = 0.25
    speed_projectile = 5
    projectile_cooldown_max = 10

    def __init__(self, pos, board_dim, player_id):
        self.id = player_id

        self.pos = pos
        self.rotation = 0
        self.board_dim = board_dim
        self.shape_size = (len(self.shape_image[0]), len(self.shape_image))

        self.projectile = Projectile(self.speed_projectile, (0, 0), board_dim)
        self.projectile_cooldown_current = 0

        self.gradient = 0
        self.x_dir = 0

    def move_look_left(self):
        self.rotation += self.speed_look

    def move_look_right(self):
        self.rotation -= self.speed_look

    def move_look_float(self, angle):
        # allows for precise control of rotation
        # angle is limited to between -1 and 1, with -1 and 1 being the maximum turn speed
        angle = 1 if angle >= 1 else angle
        angle = -1 if angle <= -1 else angle

        self.rotation += angle * self.speed_look

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
        if (check_x + self.shape_size[0] <= self.board_dim[0] and check_x >= 0 and
                check_y + self.shape_size[1] <= self.board_dim[1] and check_y >= 0):
            return True
        else:
            return False

    def move_shoot_projectile(self):
        # check if firing projectile is valid
        if self.projectile_cooldown_current <= 0:
            # set projectile position to  player position
            self.projectile.set_position(self.pos)
            # set projectile rotation to the same as the player
            self.projectile.set_rotation(self.rotation)
            # set projectile to valid
            self.projectile.valid = True
            # reset cooldown
            self.projectile_cooldown_current = self.projectile_cooldown_max

    def get_gradient_dir(self):
        # gets the current gradient of the projectile
        # convert rotation in radians to gradient
        self.gradient = math.tan(-self.rotation + math.pi/2)
        # get the x_dir using sin, round away from 0 to 1 or -1
        self.x_dir = 1 if -math.sin(self.rotation) >= 0 else -1

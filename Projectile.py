class Projectile(object):
    shape_image = [[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]]

    def __init__(self, speed, location, board_dim):
        self.speed = speed
        self.pos = list(location)
        self.direction = 0
        self.valid = False
        self.board_dim = board_dim
        self.shape_size = (len(self.shape_image[0]), len(self.shape_image))

    def set_position(self, location):
        # location given as tuple or list, sets the location of the projectile to the given location
        self.pos = list(location)

    def set_direction(self, gradient):
        # sets the direction of the projectile, using gradient
        self.direction = gradient

    def check_pos_valid(self, check_x, check_y):
        # checks if a position if within the board bounds
        if check_x + self.shape_size[0] <= 250 and check_x >= 0 and check_y + self.shape_size[1] <= 250 and check_y >= 0:
            return True
        else:
            return False

    def move_forwards(self):
        # moves the projectile forwards by self.speed for every time its called (each tick)
        new_pos_x = self.pos[0] + int(round(self.speed))
        new_pos_y = self.pos[1] + int(round(self.direction * self.speed))
        if self.valid and self.check_pos_valid(new_pos_x, new_pos_y):
            self.pos[0] = new_pos_x
            self.pos[1] = new_pos_y
        else:
            self.valid = False

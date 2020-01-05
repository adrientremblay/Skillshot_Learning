import pygame


class InputHandler(object):
    def __init__(self):
        self.player1_actions = {"forwards": False, "backwards": False, "lookleft": False, "lookright": False}
        self.player2_actions = {"forwards": False, "backwards": False, "lookleft": False, "lookright": False}

    def input_start(self, key):
        if key == pygame.K_w:
            self.player1_actions["forwards"] = True
        elif key == pygame.K_s:
            self.player1_actions["backwards"] = True
        elif key == pygame.K_a:
            self.player1_actions["lookleft"] = True
        elif key == pygame.K_d:
            self.player1_actions["lookright"] = True
        elif key == pygame.K_UP:
            self.player2_actions["forwards"] = True
        elif key == pygame.K_DOWN:
            self.player2_actions["backwards"] = True
        elif key == pygame.K_LEFT:
            self.player2_actions["lookleft"] = True
        elif key == pygame.K_RIGHT:
            self.player2_actions["lookright"] = True

    def input_stop(self, key):
        if key == pygame.K_w:
            self.player1_actions["forwards"] = False
        elif key == pygame.K_s:
            self.player1_actions["backwards"] = False
        elif key == pygame.K_a:
            self.player1_actions["lookleft"] = False
        elif key == pygame.K_d:
            self.player1_actions["lookright"] = False
        elif key == pygame.K_UP:
            self.player2_actions["forwards"] = False
        elif key == pygame.K_DOWN:
            self.player2_actions["backwards"] = False
        elif key == pygame.K_LEFT:
            self.player2_actions["lookleft"] = False
        elif key == pygame.K_RIGHT:
            self.player2_actions["lookright"] = False

    def get_inputs(self):
        return self.player1_actions, self.player2_actions

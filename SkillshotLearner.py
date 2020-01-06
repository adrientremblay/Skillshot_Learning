import numpy as np

from SkillshotGame import SkillshotGame


class SkillshotLearner(object):
    def __init__(self):
        self.actions = None
        self.model = None
        self.game = SkillshotGame()

    def model_define(self):
        # defines and creates a model
        pass

    def model_load(self):
        # loads a model from save location
        pass

    def model_train(self, epochs, mutate_threshold):
        # model plays the game and saves actions
        for epoch in range(epochs):
            # reset the game
            self.game.game_reset()
            # enter game
            while self.game.game_live:
                # get the game state

                # do actions

                # get the reward

                # save the game state and reward

                # tick the game
                self.game.game_tick()

            # after each epoch ends, fit the model
            self.model_fit()
        # after all epochs are completed
        print("Training Completed")
        # save model
        self.model_save()

    def model_fit(self, features, targets):
        # after each game is played, fit the model
        pass

    def model_save(self):
        pass

    def prepare_inputs(self, features):
        # prepares the model inputs / reshapes for model
        pass

    def calculate_reward(self, features):
        # calculates the reward from the given state
        # maximise (distance of enemy projectile to you) - (distance of your projectile to enemy)
        # -inf if hit, +inf if enemy is hit
        pass

    def plot_training(self):
        # plots the training progress
        pass

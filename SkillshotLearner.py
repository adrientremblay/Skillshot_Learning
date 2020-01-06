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

    def model_act(self, features, mutate_threshold):
        # checks threshold to see if model acts or random acts,
        # then takes features, feeds through model to find actions and performs them on model
        pass

    def model_train(self, epochs, mutate_threshold):
        # model plays the game and saves actions
        for epoch in range(epochs):
            # reset the game
            self.game.game_reset()
            # enter game
            while self.game.game_live:
                # get the game state
                game_state = self.game.get_state()
                # do actions
                self.model_act()
                # get the reward
                self.calculate_reward(game_state)
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

    def prepare_inputs(self, features, player_id):
        # prepares the model inputs / reshapes for model
        # for model trainijng against self, the dict will need to be flipped to keep consistent "self" player
        pass

    def calculate_reward(self, features, player_id):
        # calculates the reward from the given state
        # for model trainijng against self, the dict will need to be flipped to keep consistent "self" player
        # maximise (distance of enemy projectile to you) - (distance of your projectile to enemy)
        # -inf if hit, +inf if enemy is hit
        pass

    def plot_training(self):
        # plots the training progress
        pass

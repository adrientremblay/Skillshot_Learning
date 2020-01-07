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

    def model_load(self, load_epoch=-1):
        # loads a model from save location
        pass

    def model_save(self):
        # saves a model to the save location
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

    def prepare_inputs(self, features, player_id):
        # prepares the model inputs / reshapes for model
        # for model trainijng against self, the dict will need to be flipped to keep consistent "self" player
        pass

    @staticmethod
    def calculate_reward(features, player_id, opponent_id, on_target_multiplier_change=0.5):
        # calculates the reward from the given state
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        if features.get("game_winner") == player_id:
            # reward winning
            return np.inf
        elif features.get("game_winner") == opponent_id:
            # punish loosing
            return -np.inf
        elif not features.get(player_id).get("projectile_valid"):
            # punish invalid projectile
            return -np.inf
        else:
            # maximise (distance of enemy projectile to you) - (distance of your projectile to enemy)
            # add extra multiplier if the projectile is currently on target
            opponent_reward_multiplier, player_reward_multiplier = 1, 1
            if features.get(player_id).get("projectile_future_collision_opponent"):
                player_reward_multiplier -= on_target_multiplier_change
            if features.get(opponent_id).get("projectile_future_collision_opponent"):
                opponent_reward_multiplier -= on_target_multiplier_change
            return features.get(opponent_id).get("Projectile_dist_opponent") * opponent_reward_multiplier - features.get(player_id).get("projectile_dist_opponent") * player_reward_multiplier

    def plot_training(self):
        # plots the training progress
        pass

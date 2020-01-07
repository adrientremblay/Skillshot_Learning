import random

import numpy as np

from SkillshotGame import SkillshotGame


class SkillshotLearner(object):
    def __init__(self):
        self.actions = None
        self.model = None
        self.game = SkillshotGame()
        self.player_ids = (1, 2)

    def model_define(self):
        # defines and creates a model
        pass

    def model_load(self, load_epoch=-1):
        # loads a model from save location
        pass

    def model_save(self):
        # saves a model to the save location
        pass

    def model_act(self, player_id, features, mutate_threshold):
        # checks threshold to see if model acts or random acts,
        # then takes features, feeds through model to find actions and performs them on model
        np.
        return 0  # also returns the features taken

    def model_train(self, epochs, mutate_threshold):
        # model plays the game and saves actions
        for epoch in range(epochs):
            # reset the game and epoch data lists
            self.game.game_reset()
            current_epoch_progress = dict(epoch_game_state=[], epoch_actions=[])

            # enter game
            while self.game.game_live:
                # get the game state
                game_state = self.game.get_state()

                # do and save actions
                for player_id in self.player_ids:
                    current_epoch_progress.get("epoch_actions").append(self.model_act(player_id,
                                                                       self.prepare_features(game_state, player_id)[0],
                                                                       mutate_threshold))

                # save the game state - possibly also save the get_board here to visualise model later
                current_epoch_progress.get("epoch_game_state").append(game_state)

                # tick the game
                self.game.game_tick()

            # after each epoch ends, print epoch performance
            print("Epoch Completed")

            # prepare to fit by extracting rewards from epoch_game_state
            epoch_player_rewards = dict((player, []) for player in self.player_ids)
            for game_state in current_epoch_progress.get("epoch_game_state"):
                for player_id, opponent_id in zip(self.player_ids, self.player_ids[::-1]):
                    epoch_player_rewards.get(player_id).append(self.calculate_reward(game_state, player_id, opponent_id))

            # fit model
            player_features, player_targets = [], []
            for player_id in self.player_ids:
                player_features + self.prepare_features(current_epoch_progress.get("epoch_game_state"), player_id)
                player_targets + self.prepare_targets(epoch_player_rewards.get(player_id), player_id)
            self.model_fit(player_features, player_targets)  # fit can be called a single time

        # after all epochs are completed, print overall performance
        print("Training Completed")
        # save model
        self.model_save()

    def model_fit(self, features, targets, batch_size=16):
        # after each game is played, fit the model
        # shuffle features, targets and train with lower batch size for increased generalisation
        assert len(features) == len(targets)
        zipped = zip(features, targets)  # zip up to pair values
        random.shuffle(zipped)  # shuffle zipped pairs
        features, targets = zip(*zipped)  # unzip

        # fit model with batch size arg
        self.model.fit(features, targets, batch_size=batch_size, verbose=1)

    @staticmethod
    def prepare_features(features, player_id):
        # prepares the model inputs / reshapes for model
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        return [0]  # returns list

    @staticmethod
    def prepare_targets(targets, player_id):
        # prepares the model targets / reshapes for model
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        return [0]  # returns list

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

    def replay_training(self, boards):
        # takes list of boards and displays them using pygame to visualise training afterwards
        pass


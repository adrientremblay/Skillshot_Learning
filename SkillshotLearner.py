import random
import os

import numpy as np
import pandas as pd

from SkillshotGame import SkillshotGame


class SkillshotLearner(object):
    def __init__(self):
        self.model = None
        self.game = SkillshotGame()
        self.player_ids = (1, 2)

        self.save_location = "training_models"
        self.model_dir_name = "models"
        self.training_progress_dir_name = "training_progress"

    def model_define(self):
        # defines and creates a model
        pass

    def model_load(self, load_epoch=-1):
        # loads a model from save location
        pass

    def model_save(self, epochs, total_progress):
        # saves a model to the save location
        # prepare path strings
        model_save_location = self.save_location + "/" + self.model_dir_name
        progress_save_location = self.save_location + "/" + self.training_progress_dir_name

        # check if folders exist if not, create
        if not os.path.exists(model_save_location):
            os.makedirs(model_save_location)
        if not os.path.exists(progress_save_location):
            os.makedirs(progress_save_location)

        # get the epochs elapsed
        files_list = os.listdir(model_save_location)
        files_list.sort(key=lambda x: int(x.split("_"[1])))

        if len(files_list) == 0:
            epoch_start = 0
        else:
            epoch_start = files_list[-1].split("_"[1]) + 1
        epoch_end = epoch_start + epochs
        epoch_name = str(epoch_start) + "_" + str(epoch_end)

        # save model to dir with name
        self.model.save(model_save_location + "/" + epoch_name + "_model.h5")
        # save progress using pandas to_csv with append mode
        pd.DataFrame(total_progress).to_csv(progress_save_location + "/training_progress.csv", mode="a")

    def model_act(self, player_id, game_state, mutate_threshold):
        # checks threshold to see if model acts or random acts,
        # mutate threshold of 0 means all model moves, threshold of 1 means all random moves
        if np.random.rand() > mutate_threshold:
            # prepare features for the model, extract from list with length 1
            features = self.prepare_features(game_state, player_id)[0]
            # take prepared features, feed through model to find actions and performs them on model
            predictions = self.model.predict(features)
        else:
            # randomly generate actions, ensure shape is same as model output
            predictions = [0]

        # perform the generated or predicted action(s)
        self.game.get_player_by_id(player_id).move_direction_float()
        self.game.get_player_by_id(player_id).move_look_float()
        self.game.get_player_by_id(player_id).move_shoot_projectile()

        # also return the model output or the random model imitation output
        return predictions

    def model_train(self, epochs, mutate_threshold):
        # main training loop for model
        
        # create the epoch-persistent progress dicts
        total_epoch_progress = dict(epoch_ticks=[], epoch_winner=[])

        for epoch in range(epochs):
            # reset the game and epoch data lists
            self.game.game_reset()
            current_epoch_progress = dict(game_state=[], actions=[], player_rewards=dict((player, []) for player in self.player_ids))

            # enter game
            while self.game.game_live:
                # get the game state
                game_state = self.game.get_state()

                # do and save actions, model act is called twice (one for each player)
                for player_id in self.player_ids:
                    current_epoch_progress.get("actions").append(self.model_act(player_id, game_state, mutate_threshold))

                # save the game state - possibly also save the get_board here to visualise model later
                current_epoch_progress.get("game_state").append(game_state)

                # tick the game
                self.game.game_tick()

            # game ends, fitting begins
            # prepare to fit by extracting rewards from game_state in current_epoch_progress
            for game_state in current_epoch_progress.get("game_state"):
                for player_id, opponent_id in zip(self.player_ids, self.player_ids[::-1]):
                    current_epoch_progress.get("player_rewards").get(player_id).append(self.calculate_reward(game_state, player_id, opponent_id))

            # fit model
            player_features, player_targets = [], []
            for player_id in self.player_ids:
                player_features + self.prepare_features(current_epoch_progress.get("game_state"), player_id)
                player_targets + self.prepare_targets(current_epoch_progress.get("player_rewards").get(player_id), player_id)
            self.model_fit(player_features, player_targets)  # fit can be called a single time

            # move the features and targets to epoch-persistent progress dict
            total_epoch_progress.get("epoch_ticks").append(self.game.ticks)
            total_epoch_progress.get("epoch_winner").append(self.game.winner_id)

            # after each epoch ends, print epoch performance
            print("Epoch Completed, ticks taken: {}, game winner: {}".format(self.game.ticks, self.game.winner_id))

        # after all epochs are completed, print overall performance
        print("Training Completed")
        # save model and finish
        self.model_save(epochs, total_epoch_progress)
        print("model_train done.")

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
    def prepare_features(game_state, player_id):
        # prepares the model inputs / reshapes for model
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        return [0]  # returns list

    @staticmethod
    def prepare_targets(rewards, player_id):
        # prepares the model targets / reshapes for model
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        return [0]  # returns list

    @staticmethod
    def calculate_reward(game_state, player_id, opponent_id, on_target_multiplier_change=0.5):
        # calculates the reward from the given state
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        if game_state.get("game_winner") == player_id:
            # reward winning
            return np.inf
        elif game_state.get("game_winner") == opponent_id:
            # punish loosing
            return -np.inf
        elif not game_state.get(player_id).get("projectile_valid"):
            # punish invalid projectile
            return -np.inf
        else:
            # maximise (distance of enemy projectile to you) - (distance of your projectile to enemy)
            # add extra multiplier if the projectile is currently on target
            opponent_reward_multiplier, player_reward_multiplier = 1, 1
            if game_state.get(player_id).get("projectile_future_collision_opponent"):
                player_reward_multiplier -= on_target_multiplier_change
            if game_state.get(opponent_id).get("projectile_future_collision_opponent"):
                opponent_reward_multiplier -= on_target_multiplier_change
            return game_state.get(opponent_id).get("Projectile_dist_opponent") * opponent_reward_multiplier - game_state.get(player_id).get("projectile_dist_opponent") * player_reward_multiplier

    def plot_training(self):
        # plots the training progress
        pass

    def replay_training(self, boards):
        # takes list of boards and displays them using pygame to visualise training afterwards
        pass


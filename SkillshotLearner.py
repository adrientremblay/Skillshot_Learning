import random
import os

import keras
import numpy as np
import pandas as pd

from SkillshotGame import SkillshotGame


class SkillshotLearner(object):
    def __init__(self):
        self.actor = None
        self.critic = None
        self.game = SkillshotGame()
        self.player_ids = (1, 2)

        # dir locations
        self.save_location = "training_models"
        self.actor_dir_name = "actor"
        self.critic_dir_name = "critic"
        self.training_progress_dir_name = "training_progress"

        # model hyper params
        # self.model_param_mutate_threshold = 0.25  # using state space noise instead of action space noise
        self.model_param_batch_size = 16
        self.model_param_game_tick_limit = 10000

    def model_define(self):
        # defines and creates a model
        # kernel initialiser 0
        pass

    def load_actor_critic_models(self, load_index=-1):
        # loads actor and critic models from save locations
        for model_location, dir_name in zip((self.actor, self.critic), (self.actor_dir_name, self.critic_dir_name)):
            model_save_location = self.save_location + "/" + dir_name
            files_list = os.listdir(model_save_location)
            files_list.sort(key=lambda x: int(x.split("_"[1])))
            if len(files_list) > 0:
                model_location = keras.models.load_model(model_save_location + "/" + files_list[load_index])
                model_location.summary()
            else:
                print("Failed to load: ", model_save_location)
                return False
        else:
            return True

    def save_actor_critic_models(self, epochs):
        # saves actor and critic models
        for model, dir_name in zip((self.actor, self.critic), (self.actor_dir_name, self.critic_dir_name)):
            # prepare path strings
            model_save_location = self.save_location + "/" + dir_name

            # check if folders exist if not, create
            if not os.path.exists(model_save_location):
                os.makedirs(model_save_location)

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
            model.save(model_save_location + "/" + epoch_name + "_model.h5")
        print("Actor and Critic Saved.")

    def save_training_progress(self, total_progress):
        progress_save_location = self.save_location + "/" + self.training_progress_dir_name

        # check if folders exist if not, create
        if not os.path.exists(progress_save_location):
            os.makedirs(progress_save_location)

        # save progress using pandas to_csv with append mode
        pd.DataFrame(total_progress).to_csv(progress_save_location + "/training_progress.csv", mode="a")
        print("Training Progress Saved")

    def model_act(self, player_id, game_state):
        # for action space noise:
        # checks threshold to see if model acts or random acts,
        # mutate threshold of 0 means all model moves, threshold of 1 means all random moves

        # instead, use state space noise - which is added within the model (see readme)
        # prepare features for the actor, extract from list with length 1
        features = self.prepare_features(game_state, player_id)[0]
        # take prepared features, pass to actor model
        predictions = self.actor.predict(features)

        # perform the predicted action(s)
        # move_shoot_projectile is attempted every time to simply model
        self.game.get_player_by_id(player_id).move_direction_float(predictions[0])
        self.game.get_player_by_id(player_id).move_look_float(predictions[1])
        self.game.get_player_by_id(player_id).move_shoot_projectile()

        # also return the actor output
        return predictions

    def model_train(self, epochs):
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
                    current_epoch_progress.get("actions").append(self.model_act(player_id, game_state))
                # save the game state - possibly also save the get_board here to visualise model later
                current_epoch_progress.get("game_state").append(game_state)
                # tick the game
                self.game.game_tick()
                # check if the game has reached the game_tick_limit
                # useful in early training stages where the model is untrained
                if self.game.ticks == self.model_param_game_tick_limit:
                    print("Tick limit Reached.")
                    break

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
        self.save_actor_critic_models(epochs)
        self.save_training_progress(total_epoch_progress)
        print("model_train done.")

    def model_fit(self, features, targets):
        # TODO
        # after each game is played, fit the model
        # shuffle features, targets and train with lower batch size for increased generalisation
        assert len(features) == len(targets)
        zipped = zip(features, targets)  # zip up to pair values
        random.shuffle(zipped)  # shuffle zipped pairs
        features, targets = zip(*zipped)  # unzip

        # fit model with batch size arg
        self.model.fit(features, targets, batch_size=self.model_param_batch_size, verbose=1)

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
        # TODO change method to calculate the reward for both players at once, and return in list/tuple form
        # calculates the reward or q-value from the given state
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

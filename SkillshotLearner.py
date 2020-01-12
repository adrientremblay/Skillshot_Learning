import random
import os

import keras
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense, GaussianNoise, Concatenate

from SkillshotGame import SkillshotGame


class SkillshotLearner(object):
    game_state_general_keys = ["game_live",
                               "ticks",
                               "game_winner"]
    game_state_player_keys = ["player_grad",
                              "player_x_dir",
                              "player_path_dist_opponent",
                              "player_dist_opponent",
                              "player_pos_x",
                              "player_pos_y",
                              "player_rotation"]
    game_state_projectile_keys = ["projectile_cooldown",
                                  "projectile_grad",
                                  "projectile_x_dir",
                                  "projectile_path_dist_opponent",
                                  "projectile_pos_x",
                                  "projectile_pos_y",
                                  "projectile_rotation",
                                  "projectile_age",
                                  "projectile_valid",
                                  "projectile_dist_opponent",
                                  "projectile_future_collision_opponent"]

    def __init__(self):

        # environment
        self.game_environment = SkillshotGame()
        self.player_ids = (1, 2)
        self.max_dist_normaliser = (2 * (250 ** 2)) ** 0.5

        # dir locations
        self.save_location = "training_models"
        self.actor_dir_name = "actor"
        self.critic_dir_name = "critic"
        self.training_progress_dir_name = "training_progress"

        # model
        self.model_actor = None
        self.model_critic = None
        self.dim_state_space = 5
        self.dim_action_space = 2

        # model hyper-params
        # self.model_param_mutate_threshold = 0.25  # using state space noise instead of action space noise
        self.model_param_batch_size = 16
        self.model_param_game_tick_limit = 10000

    def model_define_actor(self):
        # define an actor model, which chooses actions based on the game's state

        # inputs
        state_input = Input(self.dim_state_space)

        layer_model = Dense(256, activation="relu")(state_input)
        layer_model = GaussianNoise(1.0)(layer_model)  # regularisation layer, only active during training TODO needed?
        layer_model = Dense(128, activation="relu")(layer_model)
        layer_model = GaussianNoise(1.0)(layer_model)

        # outputs
        # tanh activation is from -1 to 1, which is the correct range for the moves
        layer_output = Dense(self.dim_action_space, activation="tanh", kernel_initializer="zeros")(layer_model)

        # compile model
        actor = Model(state_input, layer_output)
        actor.compile(optimizer="adam", loss="mse")  # using default learning rate
        actor.summary()

        self.model_actor = actor

    def model_define_critic(self):
        # define a critic model, which predicts the resultant q-value from the actor's action and the game state

        # inputs
        state_input = Input(self.dim_state_space)
        actor_input = Input(self.dim_action_space)  # same shape as the actor's output layer

        layer_model = Dense(256, activation="relu")(state_input)
        layer_model = Concatenate([layer_model, actor_input])  # concat the actions with the state
        layer_model = Dense(128, activation="relu")(layer_model)

        # outputs
        # only the q-value (shape 1), and linear activation to be able to reach all q-values
        layer_output = Dense(1, activation="linear", kernel_initializer="zeros")(layer_model)

        # compile model
        critic = Model([state_input, actor_input], layer_output)
        critic.compile(optimizer="adam", loss="mse")  # using default learning rate
        critic.summary()

        self.model_critic = critic

    def load_actor_critic_models(self, load_index=-1):
        # loads actor and critic models from save locations
        for model_location, dir_name in zip((self.model_actor, self.model_critic),
                                            (self.actor_dir_name, self.critic_dir_name)):
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
        for model, dir_name in zip((self.model_actor, self.model_critic),
                                   (self.actor_dir_name, self.critic_dir_name)):
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
                epoch_start = int(files_list[-1].split("_")[1]) + 1
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

        # instead, use state space noise
        # prepare features for the actor give as list with length 1, extract from list with length 1
        features = self.prepare_states([game_state], player_id)[0]
        # take prepared features, pass to actor model
        predictions = self.model_actor.predict(features)

        # perform the predicted action(s)
        # move_shoot_projectile is attempted every time to simply model
        self.game_environment.get_player_by_id(player_id).move_direction_float(predictions[0])
        self.game_environment.get_player_by_id(player_id).move_look_float(predictions[1])
        self.game_environment.get_player_by_id(player_id).move_shoot_projectile()

        # also return the actor output
        return predictions

    def model_train(self, epochs):
        # main training loop for model

        # create the epoch-persistent progress dicts
        total_epoch_progress = dict(epoch_ticks=[], epoch_winner=[])

        for epoch in range(epochs):
            # reset the game and epoch data lists TODO add reset to random positions inside SkillshotGame
            self.game_environment.game_reset()
            cur_epoch_progress = dict(game_state=[],
                                      actions=dict((player, []) for player in self.player_ids),
                                      player_rewards=dict((player, []) for player in self.player_ids))

            # get starting state
            game_state = self.game_environment.get_state()
            cur_epoch_progress.get("game_state").append(game_state)

            # enter training loop
            while self.game_environment.game_live and self.game_environment.ticks <= self.model_param_game_tick_limit:
                # do and save actions, model act is called twice (one for each player)
                for player_id in self.player_ids:
                    cur_epoch_progress.get("actions").get(player_id).append(self.model_act(player_id, game_state))
                # tick the game
                self.game_environment.game_tick()
                # get and save the resultant game state - and possibly the get_board
                game_state = self.game_environment.get_state()
                cur_epoch_progress.get("game_state").append(game_state)

            # game ends, fitting begins

            # prepare rewards - initial state has no reward so ignore
            rewards = self.calculate_rewards(cur_epoch_progress.get("game_state")[1:])
            for reward in rewards:
                for player_id in self.player_ids:
                    cur_epoch_progress.get("player_rewards").get(player_id).append(reward.get(player_id))

            # prepare model features - final state is not used for training, so ignore
            # preparation methods split the players out
            training_states, training_actions, training_rewards = [], [], []
            for player_id in self.player_ids:
                training_states + self.prepare_states(cur_epoch_progress.get("game_state"), player_id)
                training_actions + self.prepare_actions()
                training_rewards + self.prepare_rewards()

            # fit model
            self.models_fit(training_states, training_actions, training_rewards)  # fit can be called a single time

            # move the features and targets to epoch-persistent progress dict
            total_epoch_progress.get("epoch_ticks").append(self.game_environment.ticks)
            total_epoch_progress.get("epoch_winner").append(self.game_environment.winner_id)

            # after each epoch ends, print epoch performance
            print("Epoch Completed, ticks taken: {}, game winner: {}".format(self.game_environment.ticks,
                                                                             self.game_environment.winner_id))

        # after all epochs are completed, print overall performance
        print("Training Completed")
        # save model and finish
        self.save_actor_critic_models(epochs)
        self.save_training_progress(total_epoch_progress)
        print("model_train done.")

    def models_fit(self, features, targets):
        # TODO - possibly also take in the get_state dicts
        # fits the critic model, then transfers the weights to the actor model

        # shuffle features, targets and train with lower batch size for increased generalisation
        assert len(features) == len(targets)
        zipped = zip(features, targets)  # zip up to pair values
        random.shuffle(zipped)  # shuffle zipped pairs
        features, targets = zip(*zipped)  # unzip

        # fit model with batch size arg
        # self.model.fit(features, targets, batch_size=self.model_param_batch_size, verbose=1)

    def prepare_states(self, game_states, player_id):
        # prepares the game_states for training - takes list of game states and player id
        # returns a list of trainable shape containing the states for the given player
        prepared_states = []
        for game_state in game_states:
            current_state = []
            # (general states not given to model)

            # get and normalise the player states
            current_state.append(game_state.get("player_path_dist_opponent") / self.max_dist_normaliser)
            current_state.append(game_state.get("player_dist_opponent") / self.max_dist_normaliser)
            current_state.append(game_state.get("player_pos_x") / self.game_environment.board_size[0])
            current_state.append(game_state.get("player_pos_y") / self.game_environment.board_size[1])
            current_state.append((game_state.get("player_rotation") % 2*np.pi) / 2*np.pi)

            # get and normalise the projectile states
            current_state.append(game_state.get("projectile_cooldown") / self.game_environment.get_player_by_id(player_id).projectile.cooldown_max)
            current_state.append(game_state.get("projectile_dist_opponent") / self.max_dist_normaliser)
            current_state.append(game_state.get("projectile_pos_x") / self.game_environment.board_size[0])
            current_state.append(game_state.get("projectile_pos_y") / self.game_environment.board_size[1])
            current_state.append((game_state.get("projectile_rotation") % 2*np.pi) / 2*np.pi)
            current_state.append(game_state.get("projectile_path_dist_opponent") / self.max_dist_normaliser)
            current_state.append(int(game_state.get("projectile_future_collision_opponent")))

            # append to all state lists
            prepared_states.append(current_state)

        # assert to ensure the return is the correct shape for the model
        assert prepared_states[0] == self.dim_state_space

        return prepared_states

    @staticmethod
    def prepare_actions(actions, player_id):
        # prepares the model targets / reshapes for model
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        return [0]  # returns list

    @staticmethod
    def prepare_rewards(rewards, player_id):
        # prepares the model targets / reshapes for model
        # for model training against self, the dict will need to be flipped to keep consistent "self" player
        return [0]  # returns list

    def calculate_rewards(self, game_states, on_target_multiplier_reduction=0.25, loss_reward_multiplier=2,
                          base_reward_multiplier=0.75):
        # takes a list of states and calculates the reward or q-value for each state
        # returning a list of dicts with rewards for each player

        # calculate the projectile distances for each player for each game_state beforehand
        game_states_distances = []
        for game_state in game_states:
            dist_list = [game_state.get(player_id).get("Projectile_dist_opponent") for player_id in self.player_ids]
            game_states_distances.append(dist_list)

        rewards = []
        for game_state_index, game_state in enumerate(game_states):
            state_reward = dict()
            loser_id = 0

            # first check if the game has been won
            if game_state.get("game_winner") is not 0:
                # reward winner at projectile's firing tick
                winner_id = game_state.get("game_winner")
                projectile_fired_tick = game_state_index - game_state.get(winner_id).get("projectile_age")
                rewards[projectile_fired_tick][winner_id] = 1
                # punish loosing at current tick - get loser id
                loser_id = [player_id for player_id in self.player_ids if player_id is not winner_id][0]

            for player_id, opponent_id in zip(self.player_ids, self.player_ids[::-1]):
                reward_multi = base_reward_multiplier

                # check is the projectile is currently on target
                if game_state.get(player_id).get("projectile_future_collision_opponent"):
                    # decrease multiplier of your projectile's distance
                    reward_multi = base_reward_multiplier - on_target_multiplier_reduction  # results in 0.5

                # check if player is the losing player,
                if player_id == loser_id:
                    reward_multi = base_reward_multiplier + loss_reward_multiplier  # results in 2.75

                # check if the projectile is about to expire / new projectile being fired next action 0 or 1, needs test
                if game_state.get("projectile_cooldown") == 0:
                    # get the best performance of the projectile over it's lifespan and factor in
                    start_index = game_state_index - game_state.get("projectile_age")
                    min_dist = min(game_states_distances[start_index:game_state_index])
                else:
                    min_dist = 0

                # calculate the difference of dists - +1 for indexing on dist_list - bonus for being on target
                # maximise (distance of enemy projectile to you) - (distance of your projectile to enemy)
                # also just add the minimum dist for the projectile * 2 on for projectile firing states
                # also apply reward multiplier to own projectile's distance to enemy, also divide by max_dist
                dist_list = game_states_distances[game_state_index]  # get the right pair of distances
                player_reward = (dist_list[opponent_id + 1] - (dist_list[player_id + 1] * reward_multi)) + min_dist * 2
                state_reward[player_id] = player_reward / self.max_dist_normaliser
            rewards.append(state_reward)
        return rewards

    def plot_training(self):
        # plots the training progress
        pass

    def display_training_replay(self, boards):
        # takes list of boards and displays them using pygame to visualise training afterwards
        pass

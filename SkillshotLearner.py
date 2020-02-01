import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, GaussianNoise, concatenate, Dropout

from SkillshotGame import SkillshotGame


class SkillshotLearner(object):
    model_actor: Model
    model_critic: Model
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
        self.use_random_start = True

        # dir locations
        self.save_location = "training_models"
        self.actor_dir_name = "actor"
        self.critic_dir_name = "critic"
        self.training_progress_dir_name = "training_progress"
        self.training_boards_dir_name = "training_boards"

        # model
        self.model_actor = None
        self.model_critic = None
        self.dim_state_space = 12
        self.dim_action_space = 2  # 2 continuous action inputs
        self.dim_reward_space = 1  # 1d list

        # model hyper-params
        self.model_param_batch_size = 16
        self.model_param_game_tick_limit = 2000
        self.action_noise_sd = 0.15
        self.param_noise_sd = 0.15
        # self.grad_clip_ratio = 5.0

        # tf.keras.optimizers adam optimiser, because optimizer.apply_gradients() is needed during model_actor_fit_step
        self.optimiser = tf.keras.optimizers.Adam()

    def model_define_actor(self):
        # define an actor model, which chooses actions based on the game's state

        k_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05)
        # k_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

        # inputs
        state_input = Input((self.dim_state_space,), name="state_input")

        layer_model = Dense(256, activation="relu", kernel_initializer=k_init)(state_input)
        # layer_model = GaussianNoise(1.0)(layer_model)  # regularisation layer, only active during training
        layer_model = Dense(128, activation="relu", kernel_initializer=k_init)(layer_model)
        # layer_model = GaussianNoise(1.0)(layer_model)

        # outputs
        # tanh activation is from -1 to 1, which is the correct range for the moves
        layer_output = Dense(self.dim_action_space,
                             activation="tanh",
                             kernel_initializer=k_init,
                             name="action_output")(layer_model)

        # compile model
        actor = Model(state_input, layer_output, name="actor")
        # actor.compile(optimizer="adam", loss="mse")  # using default learning rate
        actor.summary()

        self.model_actor = actor

    def model_define_critic(self):
        # define a critic model, which predicts the resultant q-value from the actor's action and the game state

        # inputs
        state_input = Input((self.dim_state_space,), name="state_input")
        actor_input = Input((self.dim_action_space,), name="action_input")  # same shape as the actor's output layer
        layer_model = Dense(256, activation="relu")(state_input)
        layer_model = Dropout(0.2)(layer_model)  # preventing overfitting of the game state
        layer_model = concatenate([layer_model, actor_input])  # concat the actions with the state
        layer_model = Dense(128, activation="relu")(layer_model)

        # outputs
        # only the q-value (shape 1), and linear activation to be able to reach all q-values
        layer_output = Dense(1,
                             activation="linear",
                             kernel_initializer="RandomNormal",
                             name="reward_output")(layer_model)

        # compile model
        critic = Model([state_input, actor_input], layer_output, name="critic")
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
                model_location = tf.keras.models.load_model(model_save_location + "/" + files_list[load_index])
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

    def load_training_progress(self):
        # load the training progress csv and returns it
        progress_save_location = self.save_location + "/" + self.training_progress_dir_name
        # TODO add check for if file exists

        return pd.read_csv(progress_save_location + "/training_progress.csv")

    def save_training_boards(self, epoch_board_list):
        boards_save_location = self.save_location + "/" + self.training_boards_dir_name
        # Todo simply saves on top ignoring pre-existing

        # check if folders exist if not, create
        if not os.path.exists(boards_save_location):
            os.makedirs(boards_save_location)

        # save progress using np.save, no append mode, so always in a new file
        array = np.array(epoch_board_list)
        np.save(boards_save_location + "/training_boards", array)
        print("Training Boards Saved")

    def load_training_boards(self):
        boards_save_location = self.save_location + "/" + self.training_boards_dir_name
        # TODO add check for if file exists
        # lines_to_read = epoch_end - epoch_start
        # assert lines_to_read > 0

        # TODO when changing save func, need to also change here so that the one to load can be specified
        # currently loads the single fie

        return np.load(boards_save_location + "/training_boards.npy", allow_pickle=True)

    def do_actions(self, player_id, predictions):
        # takes actions and actually performs them on the environment

        # perform the predicted action(s)
        self.game_environment.get_player_by_id(player_id).move_direction_float(predictions[0])
        self.game_environment.get_player_by_id(player_id).move_look_float(predictions[1])
        # move_shoot_projectile is attempted every time to simplify model
        self.game_environment.get_player_by_id(player_id).move_shoot_projectile()

    def model_act(self, game_state, player_id):
        # uses the actor to make a prediction and then acts the single given player
        # returning the action output for future training

        # prepare features for the actor give as list with length 1, extract from list with length 1
        features = self.prepare_states([game_state], player_id)[0]
        # take prepared features, pass to actor model
        predictions = self.model_actor.predict(np.expand_dims(features, 0))

        # perform the predicted action(s), exiting out of batch dim
        self.do_actions(player_id, predictions[0])

        return predictions

    def model_act_action_noise(self, game_state, player_id):
        # creates an action for the model to take, adding noise to the predicted actions / simple action noise

        # prepare features for the actor give as list with length 1, extract from list with length 1
        features = self.prepare_states([game_state], player_id)[0]
        # take prepared features, pass to actor model
        predictions = self.model_actor.predict(np.expand_dims(features, 0))

        # add noise to the predictions, which are in the range -1, 1
        predictions += np.random.normal(0, self.action_noise_sd, size=self.dim_action_space)

        # perform the predicted action(s), exiting out of batch dim
        self.do_actions(player_id, predictions[0])

        return predictions

    def model_act_param_noise(self, game_state, player_id):
        # takes the model weights, adds noise, and predicts using the modified model params
        # https://openai.com/blog/better-exploration-with-parameter-noise/

        # get the existing model weights and save for later
        existing_weights = self.model_actor.get_weights()  # self.model_actor.trainable_weights, assume trainable
        noisy_weights = [layer_w.copy() for layer_w in existing_weights]  # deep(er) copy the np.arrays

        # apply noise to noisy weights
        # TODO change to use tf.random.normal(), and set to tf.function
        for layer_weights in noisy_weights:
            # split out for future testing, but if is redundant
            if len(layer_weights.shape) == 2:  # 2d, bias (+)
                layer_weights += (layer_weights * np.random.normal(0, self.param_noise_sd, size=layer_weights.shape))
            elif len(layer_weights.shape) == 1:  # 1d, gradient / weight (*)
                layer_weights += (layer_weights * np.random.normal(0, self.param_noise_sd, size=layer_weights.shape))

        # apply the noisy weights
        self.model_actor.set_weights(noisy_weights)
        # make a prediction using the noisy weights
        features = self.prepare_states([game_state], player_id)[0]
        predictions = self.model_actor.predict(np.expand_dims(features, 0))
        # reset the weights to the original weights
        self.model_actor.set_weights(existing_weights)

        # print(self.model_actor.predict(np.expand_dims(features, 0)), "normal")
        # print(predictions, "noisy")

        # perform the predicted action(s), exiting out of batch dim
        self.do_actions(player_id, predictions[0])

        return predictions

    def model_train(self, epochs, save_progress, save_boards):
        # main training loop for model

        # create the epoch-persistent progress dicts
        total_epoch_progress = dict(epoch_ticks=[], epoch_winner=[], epoch_board_sequences=[])

        for epoch in range(epochs):
            # reset the game and epoch data lists
            self.game_environment.game_reset(random_positions=self.use_random_start)
            cur_epoch_prog = dict(game_state=[],
                                  player_actions=dict((player, []) for player in self.player_ids),
                                  player_rewards=dict((player, []) for player in self.player_ids),
                                  game_boards=[])

            # get starting state
            game_state = self.game_environment.get_state()
            cur_epoch_prog.get("game_state").append(game_state)

            # enter training loop
            while self.game_environment.game_live and self.game_environment.ticks < self.model_param_game_tick_limit:
                # do and save actions, model act is called twice (one for each player)
                for player_id in self.player_ids:
                    # player_action = self.model_act(game_state, player_id)
                    # player_action = self.model_act_action_noise(game_state, player_id)
                    player_action = self.model_act_param_noise(game_state, player_id)
                    cur_epoch_prog.get("player_actions").get(player_id).append(player_action)
                # tick the game
                self.game_environment.game_tick()
                # get and save the resultant game state
                game_state = self.game_environment.get_state()
                cur_epoch_prog.get("game_state").append(game_state)
                # get and save the game_board for future visualisation
                if save_boards:
                    cur_epoch_prog.get("game_boards").append(self.game_environment.get_board())

            # game ends, fitting preparation begins
            print("Begin Fitting for Epoch:", epoch)

            # prepare rewards - initial state has no reward so ignore
            rewards = self.calculate_rewards(cur_epoch_prog.get("game_state")[1:])
            for reward in rewards:
                for player_id in self.player_ids:
                    cur_epoch_prog.get("player_rewards").get(player_id).append(reward.get(player_id))

            # prepare model features - final state is not used for training, so ignore
            # preparation methods split the players out - ignoring the player not passed to the methods
            training_states, training_actions, training_rewards = [], [], []
            for player_id in self.player_ids:
                training_states += self.prepare_states(cur_epoch_prog.get("game_state")[:-1], player_id)
                training_actions += self.prepare_actions(cur_epoch_prog.get("player_actions"), player_id)
                training_rewards += self.prepare_rewards(cur_epoch_prog.get("player_rewards"), player_id)

            # convert to np arrays
            training_states = np.array(training_states)
            training_actions = np.array(training_actions)
            training_rewards = np.array(training_rewards)

            # assertions to ensure the training lists have been split correctly
            assert training_states.shape[0] == len(cur_epoch_prog.get("game_state")[:-1] * len(self.player_ids))
            assert training_states.shape[1] == self.dim_state_space

            actions_len = cur_epoch_prog.get("player_actions").get(1) + cur_epoch_prog.get("player_actions").get(2)
            assert training_actions.shape[0] == len(actions_len)
            assert training_actions.shape[1] == self.dim_action_space

            rewards_len = cur_epoch_prog.get("player_rewards").get(1) + cur_epoch_prog.get("player_rewards").get(2)
            assert training_rewards.shape[0] == len(rewards_len)
            assert len(training_rewards.shape) == self.dim_reward_space  # 1d list

            assert (len(cur_epoch_prog.get("game_state")) - 1) * len(self.player_ids) == training_actions.shape[0] == \
                   training_rewards.shape[0]

            # fit model, can be called a single time
            self.models_fit(training_states, training_actions, training_rewards)
            # self.models_fit(training_states, training_actions, training_rewards)

            # move the features and targets to epoch-persistent progress dict
            total_epoch_progress.get("epoch_ticks").append(self.game_environment.ticks)
            total_epoch_progress.get("epoch_winner").append(self.game_environment.winner_id)
            # also move the game board for future visualiation
            if save_boards:
                total_epoch_progress.get("epoch_board_sequences").append(cur_epoch_prog.get("game_boards"))

            # after each epoch ends, print epoch performance
            print("Epoch {} Completed, ticks taken: {}, game winner: {}".format(epoch,
                                                                                self.game_environment.ticks,
                                                                                self.game_environment.winner_id))

        # after all epochs are completed, print overall performance
        print("All Epochs Completed")

        if save_progress:
            # save model and finish
            self.save_actor_critic_models(epochs)
            self.save_training_progress(total_epoch_progress)
        if save_boards:
            self.save_training_boards(total_epoch_progress.get("epoch_board_sequences"))

    @tf.function
    def model_actor_fit_step(self, state_tensor):
        # fits the given state_tensor batch to the actor model
        # the actor fitting routine as tf.function - first time it's called it constructs a graph
        # but subsequent calls reuse the created graph

        # this is not needed, as the tf.function decorator automatically makes the input a tensor
        # state_tensor = tf.constant(state_batch, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            action_tensor = self.model_actor(state_tensor)
            critic_q_tensor = self.model_critic([state_tensor, action_tensor])

        # get the dQ/dA
        q_action_grads = tape.gradient(critic_q_tensor, action_tensor)

        # the following is to replace the below line from tf1, where a feed dict can be used when entering graph
        # every example ever calls tf.gradients() with 3 positional arguments (tf1),
        # but unable to find point in git history where there is a 3rd argument, assume grad_ys == output_gradients
        # https://stackoverflow.com/questions/42399401/use-of-grads-ys-parameter-in-tf-gradients-tensorflow
        # tf.gradients(self.model_actor.outputs, self.model_actor.trainable_weights, grad_ys=phold_grads)

        actor_training_grads = tape.gradient(action_tensor,
                                             self.model_actor.trainable_weights,
                                             output_gradients=-q_action_grads)

        # # clip gradients, also helps with param noise
        # actor_training_grads = tf.clip_by_global_norm(actor_training_grads, self.grad_clip_ratio)

        # pair up and apply the gradients to the actor's trainable weights
        grads_and_vars_to_train = zip(actor_training_grads, self.model_actor.trainable_weights)
        self.optimiser.apply_gradients(grads_and_vars_to_train)

    def models_fit(self, states, actions, rewards):
        # fits models, keeping the tensors within the graph

        assert len(states) == len(actions) == len(rewards)

        # shuffle features, targets, rewards and train with lower batch size for increased generalisation
        assert states.shape[0] == actions.shape[0] == rewards.shape[0]
        indices = np.arange(states.shape[0])
        np.random.shuffle(indices)

        states = states[indices]
        actions = actions[indices]
        rewards = rewards[indices]

        # first fit the critic
        self.model_critic.fit([states, actions], rewards, batch_size=self.model_param_batch_size, verbose=1)

        # create tf.keras.optimizers adam optimiser, because optimizer.apply_gradients() is needed
        # self.optimiser = tf.keras.optimizers.Adam()

        # fit actor in batches
        for batch_index in range(0, len(states), self.model_param_batch_size):
            state_batch = states[batch_index:batch_index + self.model_param_batch_size]
            # call the tf.function actor fitting routine
            self.model_actor_fit_step(state_batch)

    def models_fit_old(self, states, actions, rewards):
        # fits the critic model, then transfers the weights to the actor model

        # assertions to ensure inputs are correct length
        assert len(states) == len(actions) == len(rewards)

        # shuffle features, targets, rewards and train with lower batch size for increased generalisation
        assert states.shape[0] == actions.shape[0] == rewards.shape[0]
        indices = np.arange(states.shape[0])
        np.random.shuffle(indices)

        states = states[indices]
        actions = actions[indices]
        rewards = rewards[indices]

        # first fit the critic
        self.model_critic.fit([states, actions], rewards, batch_size=self.model_param_batch_size, verbose=1)

        # set up the actor fitting loop

        # get gradients of reward with respect to action in critic / effect of action on reward / dQ/dA
        # critic_action_grads = k.gradients(self.model_critic.outputs[0], self.model_critic.inputs[1])
        # place into k func so it can be called
        # critic_action_grads = k.function([self.model_critic.inputs[0], self.model_critic.inputs[1]],
        #                                  [critic_action_grads])

        # # sets up k.backend funcs to get gradients for trainable weights inside the actor
        # # k.gradients does not support grad_ys, which is why it cannot be used
        # actor_grads = k.gradients(self.model_actor.output, self.model_actor.trainable_weights)
        # actor_grads = k.function([self.model_actor.inputs], [actor_grads])
        # # calling the following will return the actor grads for the current state (when looping state)
        # actual_actor_grads = actor_grads([state])

        # create tf.keras.optimizers adam optimiser, because optimizer.apply_gradients() is needed
        optimiser = tf.keras.optimizers.Adam()

        # enter the actor training loop
        for state in states:  # can be done in batches - currently batch size is 1
            state = np.expand_dims(state, 0)  # bodge, move into for loop - expands for batch size of 1
            state_tensor = tf.constant(state, dtype=tf.float32)

            # the following is to replace the below line from tf1, where a feed dict can be used when entering graph
            # every example ever calls tf.gradients() with 3 positional arguments (tf1),
            # but am unable to find a point in git history where there is a 3rd argument
            # tf.gradients(self.model_actor.outputs, self.model_actor.trainable_weights, grad_ys=phold_grads)

            # set up the tape, stuff should be watched automatically
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                # pass the state through / make a prediction to get the values in the graph
                action_tensor = self.model_actor(state_tensor)  # not using .predict, needs to return tensor

            # get the symbolic tensor for how the critic's q value changes depending on the action
            symbolic_critic_grads = k.gradients(self.model_critic.outputs[0], self.model_critic.inputs[1])

            # tape is cleared after .gradient call, output_gradients should be equivalent to grad_ys
            # https://stackoverflow.com/questions/42399401/use-of-grads-ys-parameter-in-tf-gradients-tensorflow
            actor_training_grads = tape.gradient(action_tensor,
                                                 self.model_actor.trainable_weights,
                                                 output_gradients=-symbolic_critic_grads[0])

            # pair up the grads and weights/variables to train
            grads_and_vars_to_train = zip(actor_training_grads, self.model_actor.trainable_weights)
            # perform the gradient application, simply call in eager mode to run
            optimiser.apply_gradients(grads_and_vars_to_train)

        print("Fitting finished")

    def prepare_states(self, game_states, player_id):
        # prepares the game_states for training - takes list of game states and player id
        # returns a list of trainable shape containing the states for the given player
        # other player's states are ignored

        prepared_states = []
        for game_state in game_states:
            current_state = []
            # (general states not given to model)

            # enter player's states
            player_state = game_state.get(player_id)
            # get and normalise the player states
            current_state.append(player_state.get("player_path_dist_opponent") / self.max_dist_normaliser)
            current_state.append(player_state.get("player_dist_opponent") / self.max_dist_normaliser)
            current_state.append(player_state.get("player_pos_x") / self.game_environment.board_size[0])
            current_state.append(player_state.get("player_pos_y") / self.game_environment.board_size[1])
            current_state.append((player_state.get("player_rotation") % 2 * np.pi) / 2 * np.pi)

            # get and normalise the projectile states
            current_state.append(player_state.get("projectile_cooldown") / self.game_environment.get_player_by_id(
                player_id).projectile.cooldown_max)
            current_state.append(player_state.get("projectile_dist_opponent") / self.max_dist_normaliser)
            current_state.append(player_state.get("projectile_pos_x") / self.game_environment.board_size[0])
            current_state.append(player_state.get("projectile_pos_y") / self.game_environment.board_size[1])
            current_state.append((player_state.get("projectile_rotation") % 2 * np.pi) / 2 * np.pi)
            current_state.append(player_state.get("projectile_path_dist_opponent") / self.max_dist_normaliser)
            current_state.append(int(player_state.get("projectile_future_collision_opponent")))

            # append to all state lists
            prepared_states.append(current_state)
        return prepared_states

    @staticmethod
    def prepare_actions(actions, player_id):
        # prepares the actions taken by the players for model training
        # returns list of trainable shape containing the actions for the given player
        # other player's actions are ignored

        prepared_actions = []
        # for action in actions:
        #     prepared_actions.append(action.get(player_id))
        # assert prepared_actions.shape[0] == len(actions)

        for action in actions.get(player_id):
            prepared_actions.append(action[0])
        return prepared_actions

    @staticmethod
    def prepare_rewards(rewards, player_id):
        # prepares the rewards for actions for model training
        # returns list of trainable shape containing the rewards for the given player
        # other player's rewards are ignored

        prepared_rewards = []
        # for reward in rewards:
        #     prepared_rewards.append(reward.get(player_id))
        # assert prepared_rewards.shape[0] == len(rewards)

        for reward in rewards.get(player_id):
            prepared_rewards.append(reward)
        return prepared_rewards

    def calculate_rewards(self, game_states, on_target_multiplier_reduction=0.25, loss_reward_multiplier=2,
                          base_reward_multiplier=0.75):
        # takes a list of states and calculates the reward or q-value for each state
        # returning a list of dicts with rewards for each player

        # calculate the projectile distances for each player for each game_state beforehand
        game_states_distances = []
        for game_state in game_states:
            dist_list = [game_state.get(player_id).get("projectile_dist_opponent") for player_id in self.player_ids]
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
                opponent_index = (opponent_id + 1) % len(self.player_ids)
                player_index = (player_id + 1) % len(self.player_ids)
                player_reward = (dist_list[opponent_index] - (dist_list[player_index] * reward_multi)) + min_dist * 2
                state_reward[player_id] = player_reward / self.max_dist_normaliser
                # print(player_reward / self.max_dist_normaliser)
            rewards.append(state_reward)
        return rewards

    def plot_training(self):
        # plots the training progress
        import matplotlib.pyplot as plt

        training_progress_df = self.load_training_progress()

        training_progress_df.plot()

    def display_training_replay(self):
        # loads the boards and displays them using pygame to visualise training afterwards
        from SkillshotGameDisplay import SkillshotGameDisplay
        game_display = SkillshotGameDisplay()

        epoch_board_lists = self.load_training_boards()

        for epoch_boards in epoch_board_lists:
            game_display.display_sequence(epoch_boards)
            print("Epoch Over")


def main():
    skl = SkillshotLearner()

    skl.model_param_game_tick_limit = 100
    skl.use_random_start = True
    skl.model_define_actor()
    skl.model_define_critic()
    # skl.model_train(epochs=5, save_progress=False, save_boards=True)
    skl.model_train(epochs=5, save_progress=False, save_boards=False)

    # skl.display_training_replay()


if __name__ == "__main__":
    main()

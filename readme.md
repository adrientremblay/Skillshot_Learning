SkillshotLearner has two of the same actor-critic dueling models playing against each other.
The actor model takes the features from the SkillshotGame and outputs a parameterised action.
This action is criticised by the Critic, which attempts to predict the q-value (reward) for the parameterised actions.

More information can be found https://github.com/germain-hug/Deep-RL-Keras under Deep Deterministic Policy Gradient (DDPG)
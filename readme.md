SkillshotLearner has two of the same actor-critic dueling models playing against each other.
The actor model takes the features from the SkillshotGame and outputs a parameterised action.
This action is criticised by the Critic, which attempts to predict the q-value (reward) for the parameterised actions.

More information can be found https://github.com/germain-hug/Deep-RL-Keras under Deep Deterministic Policy Gradient (DDPG)

for reward, shift states forward and subtract prev q value
handle reward assignment problem by finding the tick that shot the winning projectile and set reward to +inf
however, loosing tick should be the tick that the projectile hits on (final tick)

for exploration, use state space noise instead of action state noise 
this means that the inputs to the model are altered instead of the outputs, leading to more effective exploration
More information can be found https://openai.com/blog/better-exploration-with-parameter-noise/
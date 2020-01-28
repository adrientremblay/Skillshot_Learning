SkillshotLearner has two of the same actor-critic dueling models playing against each other.
The actor model takes the features from the SkillshotGame and outputs a parameterised action.
This action is criticised by the Critic, which attempts to predict the q-value (reward) for the parameterised actions.

More information https://github.com/germain-hug/Deep-RL-Keras under Deep Deterministic Policy Gradient (DDPG)

for reward, shift states forward and subtract prev q value
handle reward assignment problem by finding the tick that shot the winning projectile and set reward to +inf
however, loosing tick should be the tick that the projectile hits on (final tick)
alternatively, only reward 1 on victory, -1 on loss, 0 for all other states

for exploration, use parameter noise instead of action noise 
this means that instead of adding noise to the actions after they have been predicted by the model, 
the weights of the model are altered during the prediction process, 
leading to noise that still confines to the learning of the model, instead of simply being unrelated noise
https://openai.com/blog/better-exploration-with-parameter-noise/

add planning policy - take inputs for previous [5, 10, 20] frames and process in planning model
outputting a 2d, [short, medium, long] idea representation array - which can be fed into mode actor
or planning policy can also be implemented by feeding multiple frames into actor and allowing for deep planning learning

changing the game speed variables as the model trains or after training could produce intreting results
see if model is able to adapt to different speeds
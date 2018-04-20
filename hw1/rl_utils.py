import importlib
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def load_expert_policy(name):
    policy_module = importlib.import_module('experts.' + name)
    print('loaded :3')

    return policy_module.get_env_and_policy()


class SupervisedPolicy:
    def __init__(self, env):
        input_len, output_len = env.observation_space.shape[0], env.action_space.shape[0]

        self.model = Sequential()
        self.model.add(Dense(units=64, input_dim=input_len, activation='relu'))
        self.model.add(Dense(units=output_len))


        self.model.compile(loss='mse', optimizer='sgd')

    def train(self, train_data, val_data, epochs, verbose):
        self.model.fit(train_data[0], train_data[1],
                       batch_size=128,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=val_data)

    def act(self, obs):
        obs_batch = np.expand_dims(obs, 0)
        act_batch = self.model.predict_on_batch(obs_batch)
        return np.ndarray.flatten(act_batch)

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)

def run_rollouts(env, policy, max_steps, num_rollouts):
    actions = []
    observations = []
    rewards = []

    
    for i in range(num_rollouts):
        # env, _ = load_expert_policy('roboschool-cheetah')
        obs = env.reset()
        # if i==1: print(obs)
        done = False
        reward = 0
        steps = 0

        while not done:
            act = policy.act(obs)

            actions.append(act)
            observations.append(obs)

            obs, r, done, _ = env.step(act)

            reward += r
            steps += 1
            
            if steps % 500 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        rewards.append(reward)

    return (np.array(observations), np.array(actions), np.array(rewards))

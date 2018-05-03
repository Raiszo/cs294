import gym

import importlib
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os

def load_expert_policy(name):
    policy_module = importlib.import_module('experts.' + name)
    print('loaded :3')

    return policy_module.get_env_and_policy()


class SupervisedPolicy:
    def __init__(self, observation_space, action_space):
        input_len = observation_space.shape[0] # array with 1 element -_-
        output_len = action_space.shape[0]
        
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
        # obs is a numpy array
        # add one dim to take it as a batch of size 1 :3
        obs_batch = np.expand_dims(obs, 0)
        act_batch = self.model.predict_on_batch(obs_batch)
        return np.ndarray.flatten(act_batch)

    def save(self, filename):
        self.model.save_weights(filename)

    def save_json(self):
        model_json = self.model.to_json(indent=4)
        # Save the graph
        with open('./data/model.json', 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights('./data/model.h5')
        print('saved to /data')

    def load(self, filename):
        self.model.load_weights(filename)

    def load_json(self, model_file='data/model.json', weights_file='data/model.h5'):
        # check if both of them actually exist
        assert os.path.isfile(model_file)
        assert os.path.isfile(weights_file)
        
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weights_file)

        # Compile it again
        self.model.compile(loss='mse', optimizer='sgd')


def run_rollouts(env, policy, max_steps, num_rollouts):
    actions = []
    observations = []
    rewards = []

    
    for i in range(num_rollouts):
        obs = env.reset()
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
            
            # if steps % 500 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        rewards.append(reward)

    return (np.array(observations), np.array(actions), np.array(rewards))

def get_valdata(env, policy, max_steps):
    actions = []
    observations = []

    for i in range(2):
        obs = env.reset()
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
            
            # if steps % 500 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

    return (np.array(observations)[:10], np.array(actions)[:10])

def render_demo(env, policy, steps):
    frames = []

    obs = env.reset()
    
    for i in range(steps):
        a = policy.act(obs)
        obs, r, done, _ = env.step(a)
        frames.append(env.render(mode='rgb_array'))
        if done: break

    return frames

def render_NOerrors(env_name, policy, steps):
    frames = []
    env = gym.make(env_name)

    obs = env.reset()

    for i in range(steps):
        a = policy.act(obs)
        obs, r, done, _ = env.step(a)
        frames.append(env.render(mode='rgb_array'))
        if done: break

    return frames

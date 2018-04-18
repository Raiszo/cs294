import importlib
from keras.models import Sequential
from keras.layers import Dense

def load_policy(name):
    policy_module = importlib.import_module('experts.' + name)
    print('loaded :3')

    return policy_module.get_env_and_policy()


import gym

import numpy as np
import tensorflow as tf
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process


def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):
    with tf.variable_scope(scope):
        a = input_placeholder
        for i in range(n_layers):
            a = tf.layers.dense(a, size, activation=activation, use_bias=True)

        return tf.layers.dense(a, output_size, use_bias=True)

def pathlength(path):
    return len(path["reward"])


class PolicyGradient:
    def __init__(self,
                 env_name='CartPole-v0',
                 n_iter=100, 
                 gamma=1.0, 
                 min_timesteps_per_batch=1000, 
                 max_path_length=None,
                 learning_rate=5e-3,
                 reward_to_go=True,
                 seed=0,
                 n_layers=1,
                 size=32
                 ):

        # Set random seeds
        self.seed = seed
        # tf.set_random_seed(seed)
        # np.random.seed(seed)

        # Make the gym environment
        self.env = gym.make(env_name)

        # Is this env continuous, or discrete?
        discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # Maximum length for episodes
        max_path_length = max_path_length or env.spec.max_episode_steps
        

        # Observation and action sizes
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

        """
        Placeholders observations, actions and advantages
        """
        self.sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
        self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) \
                        if discrete else \
                           tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        self.sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)

        if discrete:
            sy_logits_na = build_mlp(sy_ob_no, ac_dim, 'policy', n_layers=n_layers, size=size)
            sy_sampled_ac = tf.multinomial(sy_logits_na, 1)
            self.sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits_v2(labels=sy_ac_na, logits=sy_sampled_ac)

        self.loss = tf.reduce_mean(self.sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient.
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        if nn_baseline:
            baseline_prediction = tf.squeeze(build_mlp(
                self.sy_ob_no, 
                1, 
                "nn_baseline",
                n_layers=n_layers,
                size=size))
            # This is some actor-critic stuff, not prepared for this
            b_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=sy_b_n, predictions=baseline_prediction))
            _update_op = tf.train.AdamOptimizer(learning_rate).minimize(b_loss)

    def act(self, sess, obs):
        action = sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: obs[None]})
        return action[0]
    
    def train(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        with sess as tf.Session(config=tf_config):
            for itr in range(n_iter):
                print("********** Iteration %i ************"%itr):

                run_rollout(self.env, sess, self)

        
def run_rollout(env, sess, policy, max_steps, num_rollouts):
    # paths should be an array of size num_rollouts
    paths = []
    
    for i in range(num_rollouts) :
        obs = env.reset()
        observations, actions, rewards = [], [], []
        # animate stuff
        done = false

        for _ in range(max_steps):
            observations.append(obs)
            act = policy.act(sess, obs)[0]
            actions.append(act)

            obs, rew, done, _ = env.step(act)
            rewards.append(rew)

            if done: break

        path = {
            "observation": np.array(observations),
            "reward": np.array(rewards),
            "action": np.array(actions)
        }
        paths.append(path)

    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    re_n = np.concatenate([path["reward"] for path in paths])

    return ob_no, ac_na, re_n

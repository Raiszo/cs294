import gym

import numpy as np
import tensorflow as tf
import logz
import scipy.signal as signal
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
            a = tf.layers.dense(a, size, activation=activation, name=scope+str(i), use_bias=True)

        return tf.layers.dense(a, output_size, name=scope+'out', use_bias=True)

def pathlength(path):
    return len(path["reward"])


class PolicyGradient:
    def __init__(self,
                 env_name='CartPole-v0',
                 n_iter=100, 
                 gamma=1.0,
                 learning_rate=5e-3,
                 logdir=None,
                 max_path_length=None,
                 reward_to_go=True,
                 nn_baseline=False,
                 seed=0,
                 n_layers=1,
                 size=32
                 ):
        """
        Setup :v
        """

        self.n_iter = n_iter
        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.logger = logz
        self.logger.configure_output_dir(logdir)

        # Log experimental parameters
        args = inspect.getargspec(self.__init__)[0]
        locals_ = locals()
        params = {k: locals_[k] if k in locals_ else None for k in args}
        if params['self']: del params['self']
        # print(params)
        self.logger.save_params(params)

        
        self.reward_to_go = reward_to_go
        # Set random seeds
        self.seed = seed
        self.reward_to_go = reward_to_go
        # tf.set_random_seed(seed)
        # np.random.seed(seed)
        # Make the gym environment
        self.env = gym.make(env_name)
        # Is this env continuous, or discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Maximum length for episodes
        self.max_path_length = max_path_length or self.env.spec.max_episode_steps
        

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]

        """
        Placeholders observations, actions and advantages
        """
        self.sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="observations", dtype=tf.float32)
        self.sy_ac_na = tf.placeholder(shape=[None], name="actions", dtype=tf.int32) \
                        if discrete else \
                           tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        self.sy_re_n = tf.placeholder(shape=[None], name='rewards', dtype=tf.float32)

        if discrete:
            sy_logits_na = build_mlp(self.sy_ob_no, ac_dim, 'policy', n_layers=n_layers, size=size)
            # print(sy_logits_na.shape)
            # using multinomial sampling
            self.sy_sampled_ac = tf.one_hot(tf.multinomial(sy_logits_na, 1), ac_dim)
            # print(sy_sampled_ac.shape)
            self.sy_logprob_n = tf.losses.softmax_cross_entropy(tf.one_hot(self.sy_ac_na, ac_dim), logits=sy_logits_na)

        if nn_baseline:
            baseline_prediction = tf.squeeze(build_mlp(
                self.sy_ob_no, 
                1, 
                "nn_baseline",
                n_layers=n_layers,
                size=size))
            # self.sy_b_n = tf.placeholder(shape=[None], name='b', dtype=tf.float32)
            # b_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.sy_b_n, predictions=baseline_prediction))
            # _update_op = tf.train.AdamOptimizer(learning_rate).minimize(b_loss)
            advantage = self.sy_re_n - baseline_prediction

            if normalize_advantages: advantage = tf.nn.l2_normalize(advantage)
            
            b_loss = tf.reduce_mean(tf.losses.mean_squared_error(labesl=r_n, predictions=baseline_prediction))
        else:
            advantage = self.sy_re_n
            b_loss = 0

        pg_loss = tf.reduce_mean(advantage * self.sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient.
        # self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)




        loss = pg_loss if not nn_baseline else pg_loss + b_loss
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def act(self, sess, obs, learning_rate=5e-3):
        action = sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: obs[None]})
        return action[0]
    
    def train(self, min_steps_per_batch=1000):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        start = time.time()
        
        with tf.Session(config=tf_config) as sess:
            for itr in range(self.n_iter):
                print("********** Iteration %i ************"%itr)

                ob_no, ac_na, re_n = run_rollouts(self.env, sess, self, min_steps_per_batch, self.max_path_length, self.reward_to_go, self.gamma)
                bundle = (ob_no, ac_na, re_n)
                
                self.update_policy(sess, bundle)
                returns = [path["reward"].sum() for path in paths]
                ep_lengths = [pathlength(path) for path in paths]
                self.logger.log_tabular("Time", time.time() - start)
                self.logger.log_tabular("Iteration", itr)
                self.logger.log_tabular("AverageReturn", np.mean(returns))
                self.logger.log_tabular("StdReturn", np.std(returns))
                self.logger.log_tabular("MaxReturn", np.max(returns))
                self.logger.log_tabular("MinReturn", np.min(returns))
                self.logger.log_tabular("EpLenMean", np.mean(ep_lengths))
                self.logger.log_tabular("EpLenStd", np.std(ep_lengths))
                self.logger.log_tabular("TimestepsThisBatch", timesteps_this_batch)
                self.logger.log_tabular("TimestepsSoFar", total_timesteps)
                self.logger.dump_tabular()
                self.logger.pickle_tf_vars()

    def upadte_policy(self, sess, bundle):
        ob_no, ac_na, re_n = bundle
        feed_dict = {
            self.sy_ob_no: ob_no,
            self.sy_ac_na: ac_na,
            self.sy_re_n: re_n
        }
        sess.run(self.train_op, feed_dict=feed_dict)
        
def run_rollouts(env,
                 sess,
                 policy,
                 min_steps_per_batch,
                 max_path_length,
                 reward_to_go=False,
                 gamma=1):
    # paths should be an array of size num_rollouts
    paths = []
    steps_total = 0
    num_rollouts = 0
    
    while steps_total < min_steps_per_batch:
        obs = env.reset()
        observations, actions, rewards = [], [], []
        # animate stuff
        done = False

        while not done or actions.length < max_path_length:
            observations.append(obs)
            act = policy.act(sess, obs)[0]
            actions.append(act)

            obs, rew, done, _ = env.step(act)
            rewards.append(rew)

            if done or actions.length >= max_path_length : break

        path = {
            "observation": np.array(observations),
            "reward": np.array(rewards),
            "action": np.array(actions)
        }
        paths.append(path)

        steps_total += path.length
        num_rollouts += 1

    print(num_rollouts)
    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    if reward_to_go == False:
        re_n = np.concatenate([ discounted(path['reward'], gamma)[0]*np.ones_like(path['actions']) for path in paths ])
    else:
        re_n = np.concatenate([ discounted(path['reward'], gamma) for path in paths ])
        
    return ob_no, ac_na, re_n

def discounted(re, discount):
    r = re[::-1]
    a = [1, -discount]
    b = [1]

    return signal.lfilter(b, a, x=r)[::-1]

def wrapper(args):
    """
    Put everything inside a single function, for a high amount of experiments
    Just like cs294, first call the PolicyGradient constructor with args
    then train :3
    """
    return 0

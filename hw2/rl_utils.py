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
                 exp_name,
                 env_name='CartPole-v0',
                 n_iter=100, 
                 gamma=1.0,
                 learning_rate=5e-3,
                 logdir=None,
                 max_path_length=None,
                 reward_to_go=True,
                 normalize_advantages=True,
                 nn_baseline=False,
                 seed=0,
                 n_layers=1,
                 size=32
                 ):
        """
        Setup :v
        """

        self.exp_name = exp_name
        self.n_iter = n_iter
        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.logger = logz
        self.logger.configure_output_dir(logdir)
        # Set random seeds
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)
        
        # Log experimental parameters
        args = inspect.getargspec(self.__init__)[0]
        locals_ = locals()
        params = {k: locals_[k] if k in locals_ else None for k in args}
        if params['self']: del params['self']
        # print(params)
        self.logger.save_params(params)

        
        # Make the gym environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        # Is this env continuous, or discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.discrete = discrete
        # Maximum length for episodes
        self.max_path_length = max_path_length or self.env.spec.max_episode_steps 
        

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]

        """
        Placeholders observations, actions and advantages
        """
        self.sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="observations", dtype=tf.float32)
        # If this is discrete, actions are just one number: 0,1,2 ... ac_dim-1
        self.sy_ac_na = tf.placeholder(shape=[None], name="actions", dtype=tf.int32) \
                        if discrete else \
                           tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        # Define a placeholder for Value of a state
        self.sy_q_n = tf.placeholder(shape=[None], name='rewards', dtype=tf.float32)
        # self.sy_target_n = tf.placeholder(shape=[None], name='baseline prediction', dtype=tf.float32)

        if discrete:
            sy_logits_na = build_mlp(self.sy_ob_no, ac_dim, 'policy', n_layers=n_layers, size=size)
            # using multinomial sampling, because this is a stochastic policy
            # This is the sampled action from the policy
            self.sy_sampled_ac = tf.multinomial(sy_logits_na - tf.reduce_max(sy_logits_na, axis=1, keepdims=True), 1)
            self.sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_ac_na, logits=sy_logits_na)
        else:
            sy_mean = build_mlp(self.sy_ob_no, ac_dim, 'policy',
                                n_layers=n_layers, size=size)
            sy_logstd = tf.Variable(tf.zeros([1, ac_dim]), name='logstd',
                                    dtype=tf.float32)
            sy_std = tf.exp(sy_logstd)
            self.sy_sampled_ac = sy_mean + sy_std \
                                 * tf.random_normal(tf.shape(sy_mean))
            sy_z = (self.sy_ac_na - sy_mean) / sy_std
            self.sy_logprob_n = -0.5* tf.reduce_sum(tf.square(sy_z), axis=1)
            
        # TODO the non discrete case :P
        

        eps = 1e-8
        if nn_baseline:
            # baseline_prediction should always compute normalized values
            baseline_prediction = tf.squeeze(build_mlp(
                self.sy_ob_no, 
                1, 
                "nn_baseline",
                n_layers=n_layers,
                size=size))
            # Normalize the placeholder that is feed for this operation
            [mean_qn, var_qn] = tf.nn.moments(self.sy_q_n, axes=[0])
            std_qn = tf.sqrt(var_qn)
            
            baseline_target = (self.sy_q_n - mean_qn) / (std_qn + eps)
            b_loss = tf.reduce_mean(tf.square(baseline_target - baseline_prediction))

            # Rescale the prediction to match the current q_n mean and variance
            baseline_rescaled = baseline_prediction * std_qn + mean_qn
            adv_n = self.sy_q_n - baseline_rescaled
        else:
            b_loss = 0
            adv_n = self.sy_q_n

        if normalize_advantages:
            [mean_adv, var_adv] = tf.nn.moments(adv_n, axes=[0])
            std_adv = tf.sqrt(var_adv)
            adv_n = (adv_n - mean_adv) / (std_adv + 1e-8)

        pg_loss = tf.reduce_mean(adv_n * self.sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient.
        # self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



        # Hardcoded
        gradient_clip = 40

        self.loss = pg_loss + b_loss
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = tf.gradients(self.loss, tf.trainable_variables())
        self.grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
        grads_and_vars = list(zip(self.grads, tf.trainable_variables()))
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        # self.train_op = optimizer.minimize(self.loss)

    def act(self, sess, obs, learning_rate=5e-3):
        action = sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: obs[None]})
        if self.discrete:
            # return a single number
            return action[0][0]
        else:
            # return a vector
            return action[0]
    
    def update_policy(self, sess, bundle):
        ob_no, ac_na, re_n = bundle
        feed_dict = {
            self.sy_ob_no: ob_no,
            self.sy_ac_na: ac_na,
            self.sy_q_n: re_n
        }
        sess.run(self.train_op, feed_dict=feed_dict)
        # sess.run(self.train_op, feed_dict=feed_dict)
        # sess.run(self.train_op, feed_dict=feed_dict)
        # print(ac)

    def train(self, print_console=False, steps_eg=100, batch_size=1000):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        start = time.time()
        
        with tf.Session(config=tf_config) as sess:
            # print(tf.trainable_variables())
            sess.run(tf.global_variables_initializer()) #pylint: disable=E1101
            total_timesteps = 0
            for itr in range(self.n_iter):
                if itr % 20 == 0 or itr == self.n_iter - 1:
                    print("********** Iteration %i ************"%itr)

                ob_no, ac_na, re_n, returns, ep_lengths, timesteps = run_rollouts(self.env, sess, self, batch_size, self.max_path_length, self.reward_to_go, self.gamma)
                bundle = (ob_no, ac_na, re_n)
                total_timesteps += timesteps
                self.update_policy(sess, bundle)
                
                self.logger.log_tabular("Time", time.time() - start)
                self.logger.log_tabular("Iteration", itr)
                self.logger.log_tabular("AverageReturn", np.mean(returns))
                self.logger.log_tabular("StdReturn", np.std(returns))
                self.logger.log_tabular("MaxReturn", np.max(returns))
                self.logger.log_tabular("MinReturn", np.min(returns))
                self.logger.log_tabular("EpLenMean", np.mean(ep_lengths))
                self.logger.log_tabular("EpLenStd", np.std(ep_lengths))
                self.logger.log_tabular("TimestepsThisBatch", timesteps)
                self.logger.log_tabular("TimestepsSoFar", total_timesteps)
                self.logger.dump_tabular(print_console)
                self.logger.pickle_tf_vars()

            frames = []
            env = gym.make(self.env_name)
            obs = env.reset()

            for i in range(steps_eg):
                a = self.act(sess, obs)
                obs, r, done, _ = env.step(a)
                frames.append(env.render(mode='rgb_array'))
                if done: break
                
            return frames
        
def run_rollouts(env,
                 sess,
                 policy,
                 batch_size,
                 max_path_length,
                 reward_to_go,
                 gamma=1):
    # paths should be an array of size num_rollouts
    paths = []
    steps_total = 0
    num_rollouts = 0
    
    while steps_total < batch_size:
        obs = env.reset()
        observations, actions, rewards = [], [], []
        # animate stuff
        done = False

        while not done and len(actions) < max_path_length:
            observations.append(obs)
            act = policy.act(sess, obs)
            actions.append(act)

            obs, rew, done, _ = env.step(act)
            rewards.append(rew)

            # if done or len(actions) >= max_path_length : break

        path = {
            "observation": np.array(observations),
            "reward": np.array(rewards),
            "action": np.array(actions)
        }
        paths.append(path)

        steps_total += len(rewards)
        num_rollouts += 1

    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    if reward_to_go == False:
        re_n = np.concatenate([ discounted(path['reward'], gamma)[0]*np.ones_like(path['actions']) for path in paths ])
    else:
        re_n = np.concatenate([ discounted(path['reward'], gamma) for path in paths ])
    sum_re_n = np.array([path['reward'].sum() for path in paths])
    ep_lengths = np.array([len(path['reward']) for path in paths])
    timesteps = steps_total
        
    return ob_no, ac_na, re_n, sum_re_n, ep_lengths, timesteps

def discounted(re, discount):
    r = re[::-1]
    a = [1, -discount]
    b = [1]

    return signal.lfilter(b, a, x=r)[::-1]

def args_jupyter(args, seed):
    """
    When using jupyter, arguments are entered as a dictionary :3
    """
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.get('exp_name', 'vpg') + '_' + args.get('env_name') + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    return {
        'exp_name': args.get('exp_name', 'vpg'),
        'env_name': args.get('env_name'),
        'n_iter': args.get('n_iter', 100),
        'gamma': args.get('gamma', 1.0),
        'learning_rate': args.get('learning_rate', 5e-3),

        'logdir': os.path.join(logdir, '%d'%seed),

        'max_path_length': args.get('max_path_length', None),
        'reward_to_go': args.get('reward_to_go', True),
        'normalize_advantages': args.get('normalize_advantages', False),
        'nn_baseline': args.get('nn_baseline', False),
        'batch_size': args.get('batch_size', 1000),

        'seed': seed,

        'n_layers': args.get('n_layers', 1),
        'size': args.get('size', 32)
    }

def wrapper(args, logdir):
    """
    Put everything inside a single function, for a high amount of experiments
    Just like cs294, first call the PolicyGradient constructor with args
    then train :3
    """

    PG = PolicyGradient(
        exp_name=args.get('exp_name'),
        env_name=args.get('env_name'),
        n_iter=args.get('n_iter'),
        gamma=args.get('gamma'),
        learning_rate=args.get('learning_rate'),
        logdir=args.get('logdir'),
        max_path_length=args.get('max_path_length'),
        reward_to_go=args.get('reward_to_go'),
        normalize_advantages=args.get('normalize_advantages'),
        nn_baseline=args.get('nn_baseline'),
        seed=args.get('seed'),
        n_layers=args.get('n_layers'),
        size=args.get('size')
    )

    PG.train(print_console=False, batch_size=args.get('batch_size'))

def render_NOerrors(env_name, policy, steps):
    frames = []
    env = gym.make(env_name)
    
    with tf.Session() as sess:
        obs = env.reset()

        for i in range(steps):
            a = policy.act(sess, obs)
            obs, r, done, _ = env.step(a)
            frames.append(env.render(mode='rgb_array'))
            if done: break

        return frames

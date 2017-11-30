# -*- coding: utf-8 -*-

'''
    This module defines QNetwork agent which uses Q-learning with Q-network.
'''

from gym import spaces
from agent import Agent

import numpy as np
import tensorflow as tf

from util import get_tf_config


class QNetwork(Agent):
    '''
        This class defines an Agent which uses Q-learning with
        state-action network.
        Params of __init__:
            - env: Environment -- environment to use;
            - gamma: float -- discount factor;
            - learning_rate: float -- learning rate.
            - num_units: int -- number of units in layer
            - num_layers: int -- number of layers
    '''

    def __init__(self, env,
                 gamma=0.99,
                 learning_rate=0.1,
                 num_units=1,
                 num_layers=0):
        super(QNetwork, self).__init__(env)

        if isinstance(env.observation_space, spaces.Discrete):
            self.num_states = env.observation_space.n
            self.is_discrete = True
        else:
            self.state_shape = env.observation_space.shape
            self.is_discrete = False

        self.num_actions = env.action_space.n

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = 0.1

        self.num_layers = num_layers
        self.num_units = num_units

        tf.reset_default_graph()
        self.build()
        self.sess = tf.Session(graph=self.graph, config=get_tf_config())

        self.sess.run(self.init)


    def build(self):
        '''
            This function builds TF graph and all the ops
            belonging to it. As a result new members are acquired:
                - self.out: tensor [batch_size, num_actions]
                    Q(s, a) for all actions
                - self.predict: : predicted action for greedy policy
                - self.update: train_op -- updates neural network using Bellman equation
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            if self.is_discrete:
                state_shape = [1, self.num_states]
            else:
                state_shape = [1] + list(self.state_shape)

            self.state_placeholder = tf.placeholder(shape=state_shape, dtype=tf.float32)

            out = self.state_placeholder

            for i in range(self.num_layers):
                out = tf.layers.dense(out,
                                      self.num_units,
                                      name="dense_{}".format(i),
                                      activation=tf.nn.relu)

            out = tf.layers.dense(out, self.num_actions, name="logits",
                                  activation=None,
                                  use_bias=False,
                                  kernel_initializer=tf.random_uniform_initializer(
                                      minval=0,
                                      maxval=0.01))

            self.out = out
            self.predict = tf.argmax(out, axis=1)


            old_Q = out
            self.new_Q = tf.placeholder(shape=[1, self.num_actions], dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.square(self.new_Q - old_Q))
            self.update = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            self.init = tf.global_variables_initializer()


    def compute_Q(self, observation):
        '''
            Helper method to compute Q values at given state.
            Params:
                - observation: State -- state at which Q-values are computed
            Returns:
                - Q: Tensor [batch_size x num_actions] -- Q-values
        '''
        observation = self.preprocess(observation)[None]

        Q = self.sess.run(self.out,
                          feed_dict={self.state_placeholder: observation})

        return Q


    def preprocess(self, observation):
        '''
            This function does preprocessing for discrete observations.
            Params:
                - observation: State -- state to be preprocessed
            Returns:
                - out: State -- one-hot encoded state if discrete and the same state if not
        '''
        if self.is_discrete:
            out = np.identity(self.num_states)[observation]

        return out


    def observe(self, old_observation, action, new_observation, reward, done):
        Q1 = self.compute_Q(new_observation)
        max_Q1 = np.max(Q1)

        targetQ = self.compute_Q(old_observation)
        targetQ[0, action] = reward + self.gamma * max_Q1

        self.sess.run(self.update,
                      feed_dict={
                          self.state_placeholder:self.preprocess(old_observation)[None],
                          self.new_Q:targetQ
                          })


    def act(self, observation):
        self.step_num += 1

        action = self.sess.run(self.predict,
                               feed_dict={self.state_placeholder:self.preprocess(observation)[None]})

        if np.random.rand(1) < self.eps:
            action[0] = self.env.action_space.sample()

        return action[0]


    def episode_end(self):
        self.eps = 1. / (10 + self.episode_num // 50)
        super(QNetwork, self).episode_end()

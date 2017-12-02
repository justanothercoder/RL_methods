# -*- coding: utf-8 -*-

'''
    This module defines Reinforce agent which uses REINFORCE algorithm.
'''

from gym import spaces
from agent import Agent
from memory import Memory

import numpy as np
import tensorflow as tf

from util import get_tf_config
from feedforward_policy import FeedForwardPolicy


def _one_hot(k, n):
    out = np.zeros(n)
    out[k] = 1
    return out


class Reinforce(Agent):
    '''
        This class defines an Agent which uses Q-learning with
        state-action network.
        Params of __init__:
            - env: Environment -- environment to use;
            - gamma: float -- discount factor;
            - learning_rate: float -- learning rate.
            - num_units: int -- number of units in layer
            - num_layers: int -- number of layers
            - update_frequency: int -- number of episodes per update
    '''

    def __init__(self, env,
                 gamma=0.99,
                 learning_rate=0.1,
                 num_units=1,
                 num_layers=0,
                 update_frequency=5):
        super(Reinforce, self).__init__(env)

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency

        self.num_units = num_units
        self.num_layers = num_layers

        self.memory = Memory()

        tf.reset_default_graph()

        self.build()
        self.sess = tf.Session(config=get_tf_config())

        self.sess.run(self.init)


    def build(self):
        '''
            This function builds TF graph and all the ops
            belonging to it. As a result new members are acquired:
                - self.out: tensor [batch_size, action_shape]
                    action or their logits
                - self._state: state placeholder
                - self._action: action placeholder
                - self._reward: reward placeholder
                - self.loss: loss tensor
                - self.update: train_op -- updates neural network using REINFORCE
                - self.init: all variables initializer
        '''
        def num_or_shape(space):
            return space.n if isinstance(space, spaces.Discrete) else space.shape

        state_num_or_shape = num_or_shape(self.env.observation_space)
        action_num_or_shape = num_or_shape(self.env.action_space)

        self.policy = FeedForwardPolicy(self.num_layers,
                                        self.num_units,
                                        state_num_or_shape,
                                        action_num_or_shape)

        self._state = self.policy.state
        self._action = self.policy.action
        self._reward = tf.placeholder(shape=[None], dtype=tf.float32)

        self.loss = -tf.reduce_mean(self._reward * self.policy.log_probability)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.update = optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()


    def preprocess_state(self, observation):
        '''
            This function does preprocessing for discrete observations.
            Params:
                - observation: State -- state to be preprocessed
            Returns:
                - out: State -- one-hot encoded state if discrete and the same state if not
        '''
        if self.policy.discrete_states:
            observation = _one_hot(observation, self.num_states)

        return observation


    def observe(self, old_observation, action, new_observation, reward, done):
        old_observation = self.preprocess_state(old_observation)
        new_observation = self.preprocess_state(new_observation)

        self.memory.insert(old_observation,
                           action,
                           new_observation,
                           reward,
                           done)

        if done and (self.episode_num + 1) % self.update_frequency == 0:
            discounted_rewards = self.memory.compute_returns(self.gamma)

            self.sess.run(self.update,
                          feed_dict={
                              self._state: self.memory.old_states,
                              self._action: self.memory.actions,
                              self._reward: discounted_rewards
                              })


    def act(self, observation):
        observation = self.preprocess_state(observation)
        return self.sess.run(self.policy.sample,
                             feed_dict={self._state: [observation]})[0]


    def episode_end(self):
        if (self.episode_num + 1) % self.update_frequency == 0:
            self.memory.clear()

# -*- coding: utf-8 -*-

'''
    This module defines FeedForwardPolicy. It is modelling
    policy which takes state as input and returns distribution
    over actions given that state.
'''

from policy import Policy

import tensorflow as tf
from util import fc_network


class FeedForwardPolicy(Policy):
    '''
        This class defines FeedForwardPolicy.
        It is modelling policy as neural network, particularly
        Multi-layer perceptron (MLP) with normal or categorical
        distribution at the end.

        Params of __init__:
            - num_layers: int -- number of layers of MLP
            - num_units: int -- number of units in each layer
            - state_num_or_shape: int or tuple
                * if states are discrete -- this is the number of states
                * if not -- this is the shape of state
            - action_num_or_shape: int or tuple
                * if actions are discrete -- this is the number of actions
                * if not -- this is the shape of action
    '''

    def __init__(self, num_layers, num_units, state_num_or_shape, action_num_or_shape):
        self.num_layers = num_layers
        self.num_units = num_units

        if isinstance(state_num_or_shape, int):
            self.num_states = state_num_or_shape
            self.state_shape = [self.num_states]
            self.discrete_states = True
        else:
            self.state_shape = state_num_or_shape
            self.discrete_states = False

        if isinstance(action_num_or_shape, int):
            self.num_actions = action_num_or_shape
            self.action_shape = []
            self.discrete_actions = True
        else:
            self.action_shape = action_num_or_shape
            self.discrete_actions = False

        self.build()


    def build(self):
        '''
            This method builds all ops relevant to the
            policy. Among them:
                - self._state -- state placeholder
                - self._action -- action placeholder
                - self._predict -- greedy prediction for action
                - self._stoch_predict -- non-greedy prediction for action
                - self._logprob -- log probability of action given state
        '''

        state_shape = [None] + list(self.state_shape)
        action_shape = [None] + list(self.action_shape)

        self._state = tf.placeholder(shape=state_shape, dtype=tf.float32, name="state")
        self._action = tf.placeholder(shape=action_shape, dtype=tf.float32, name="action")

        self._pre_out = fc_network(self._state, self.num_layers, self.num_units)
        self._out = tf.layers.dense(self._pre_out, self.num_actions, activation=None)

        if self.discrete_actions:
            cat = tf.distributions.Categorical(logits=self._out)

            self._predict = cat.mode()
            self._stoch_predict = cat.sample()
            self._logprob = cat.log_prob(self._action)
        else:
            self._mu, self._log_sigma = tf.split(self._out, 2, axis=1)

            norm = tf.distributions.Normal(self._mu, tf.exp(self._log_sigma))

            self._predict = norm.mean()
            self._stoch_predict = norm.sample()
            self._logprob = norm.log_prob(self._action)


    @property
    def predict(self):
        return self._predict


    @property
    def sample(self):
        return self._stoch_predict


    @property
    def log_probability(self):
        return self._logprob


    @property
    def state(self):
        return self._state


    @property
    def action(self):
        return self._action

# -*- coding: utf-8 -*-

'''
    This module defines FeedForwardPolicy. It is modelling
    policy which takes state as input and returns distribution
    over actions given that state.
'''

from feedforward_policy import FeedForwardPolicy

import tensorflow as tf
from util import fc_network


class FeedForwardActorCritic(FeedForwardPolicy):
    '''
        This class defines FeedForwardActorCritic.
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

        super(FeedForwardActorCritic, self).build()
        
        with tf.variable_scope('critic'):
            self._pre_out_value = fc_network(self._state, self.num_layers, self.num_units)
            self._value_pred = tf.layers.dense(self._pre_out_value, 1)[:, 0]


    @property
    def value_pred(self):
        return self._value_pred

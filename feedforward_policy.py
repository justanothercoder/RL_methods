# -*- coding: utf-8 -*-

from policy import Policy

import tensorflow as tf
from util import fc_network


class FeedForwardPolicy(Policy):
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
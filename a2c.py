# -*- coding: utf-8 -*-

'''
    This module defines A2C agent which uses REINFORCE algorithm
    with actor-critic.
'''

from gym import spaces
from agent import Agent
from memory import Memory

import numpy as np
import tensorflow as tf

from util import get_tf_config
from feedforward_actor_critic import FeedForwardActorCritic


def _one_hot(k, n):
    out = np.zeros(n)
    out[k] = 1
    return out


class A2C(Agent):
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
                 lambd=0.7,
                 learning_rate=0.1,
                 num_units=1,
                 num_layers=0,
                 update_frequency=5):
        super(A2C, self).__init__(env)

        self.gamma = gamma
        self.lambd = lambd
        
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

        self.actor_critic = FeedForwardActorCritic(self.num_layers,
                                                   self.num_units,
                                                   state_num_or_shape,
                                                   action_num_or_shape)

        self._state = self.actor_critic.state
        self._action = self.actor_critic.action
        self._advantage = tf.placeholder(shape=[None], dtype=tf.float32)
        self._value_target = tf.placeholder(shape=[None], dtype=tf.float32)

        self.policy_loss = -tf.reduce_mean(self._advantage * self.actor_critic.log_probability)
#        self.policy_loss = -tf.reduce_mean((self._advantage - self.actor_critic.value_pred) * self.actor_critic.log_probability)
        self.value_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.actor_critic.value_pred, self._value_target))

        self.loss = self.policy_loss + self.value_loss

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
        if self.actor_critic.discrete_states:
            E = np.identity(self.actor_critic.num_states)
            observation = E[observation]            

        return observation


    def observe(self, old_observation, action, new_observation, reward, done, value_pred=None):
        old_observation = self.preprocess_state(old_observation)
        new_observation = self.preprocess_state(new_observation)

        self.memory.insert(old_observation,
                           action,
                           new_observation,
                           reward,
                           done,
                           value_pred)

        self.next_pred = self.sess.run(self.actor_critic.value_pred,
                                       feed_dict={self._state: new_observation}) 

        
    def act(self, observation):
        observation = self.preprocess_state(observation)
        action, value = self.sess.run([
            self.actor_critic.sample, 
            self.actor_critic.value_pred
            ],
            feed_dict={self._state: observation})
        
        return action, value


    def episode_end(self):
        if (self.episode_num + 1) % self.update_frequency == 0:
            advantages, returns = self.memory.compute_advantages(
                    self.gamma, 
                    self.lambd, 
                    self.next_pred)

#            returns = self.memory.compute_returns(self.gamma)
            
            states = self.memory.old_states.reshape(-1, *self.actor_critic.state_shape)
            actions = self.memory.actions.reshape(-1, *self.actor_critic.action_shape)

            self.sess.run(self.update,
                          feed_dict={
                              self._state: states,
                              self._action: actions,
                              self._advantage: advantages.reshape(-1),
                              self._value_target: returns.reshape(-1)
                              })

#            self.sess.run(self.update,
#                          feed_dict={
#                              self._state: states,
#                              self._action: actions,
#                              self._advantage: returns.reshape(-1),
#                              self._value_target: returns.reshape(-1)
#                              })


            self.memory.clear()
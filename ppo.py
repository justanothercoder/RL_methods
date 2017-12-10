# -*- coding: utf-8 -*-

'''This module defines PPO agent which uses PPO algorithm.'''

from gym import spaces
from a2c import A2C

import numpy as np
import tensorflow as tf

from feedforward_actor_critic import FeedForwardActorCritic

def _one_hot(k, n):
    out = np.zeros(n)
    out[k] = 1
    return out


class PPO(A2C):
    '''
        This class defines an Agent which uses A2C + PPO.
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
                 update_frequency=5, clip_param=0.1,
                 normalize_advantages=True, ppo_epochs=1,
                 batch_size=64):

        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.normalize_advantages = normalize_advantages
        self.batch_size = batch_size

        super(PPO, self).__init__(env,
             gamma,
             lambd,
             learning_rate,
             num_units,
             num_layers,
             update_frequency)


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
                - self.update: train_op -- updates neural network using PPO
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
        self._old_action_log_probability = tf.placeholder(shape=[None], dtype=tf.float32)

        ratio = tf.exp(self.actor_critic.log_probability - self._old_action_log_probability)
        clipped_ratio = tf.clip_by_value(ratio, 1. - self.clip_param, 1. + self.clip_param)

        policy_loss = tf.minimum(self._advantage * ratio,
                                 self._advantage * clipped_ratio)
        self.policy_loss = -tf.reduce_mean(policy_loss)

        self.value_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.actor_critic.value_pred, self._value_target))
        self.loss = self.policy_loss + self.value_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.update = optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()


    def observe(self, old_observation, action, new_observation, reward, done, value_pred=None):
        old_observation = self.preprocess_state(old_observation)
        new_observation = self.preprocess_state(new_observation)

        action_logprob = self.sess.run(self.actor_critic.log_probability,
                                       feed_dict={
                                               self._state: old_observation,
                                               self._action: action})

        self.memory.insert(old_observation,
                           action,
                           new_observation,
                           reward,
                           done,
                           value_pred,
                           action_logprob)

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

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

#            returns = self.memory.compute_returns(self.gamma)

            for i in range(self.ppo_epochs):

                for batch in self.memory.batch_generator(self.batch_size):
                    states = batch.old_states.reshape(-1, *self.actor_critic.state_shape)
                    actions = batch.actions.reshape(-1, *self.actor_critic.action_shape)
                    old_action_log_probability = batch.old_action_log_probability.reshape(-1)

                    self.sess.run(self.update,
                                  feed_dict={
                                      self._state: states,
                                      self._action: actions,
                                      self._advantage: batch.advantages.reshape(-1),
                                      self._value_target: batch.returns.reshape(-1),
                                      self._old_action_log_probability: old_action_log_probability
                                      })

            self.memory.clear()
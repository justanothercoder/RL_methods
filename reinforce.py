# -*- coding: utf-8 -*-

'''
    This module defines Reinforce agent which uses REINFORCE algorithm.
'''

from gym import spaces
from agent import Agent
from memory import Memory

import numpy as np
import tensorflow as tf

from util import get_tf_config, fc_network


def _placeholder(shape, name=None, dtype=tf.float32):
    return tf.placeholder(dtype, shape, name)


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
    '''

    def __init__(self, env,
                 gamma=0.99,
                 learning_rate=0.1,
                 num_units=1,
                 num_layers=0,
                 update_frequency=5):
        super(Reinforce, self).__init__(env)
        self.adapt_structure_to_env(env)

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency

        self.num_layers = num_layers
        self.num_units = num_units

        self.memory = Memory()

        tf.reset_default_graph()
        self.build()
        self.sess = tf.Session(graph=self.graph, config=get_tf_config())

        self.sess.run(self.init)


    def adapt_structure_to_env(self, env):
        '''
            This method looks at the environment
            and determines if states and actions are discrete
            or not. If they are they should be pre- or post- processed.

            Params:
                - env: Environment -- environment to adapt to.
            Returns:
                - set self.discrete_states iff states are discrete
                - set self.discrete_actions iff actions are discrete
                - set self.num_states or self.state_shape
                - set self.num_actions or self.action_shape
        '''
        if isinstance(env.observation_space, spaces.Discrete):
            self.num_states = env.observation_space.n
            self.state_shape = [self.num_states]
            self.discrete_states = True
        else:
            self.state_shape = env.observation_space.shape
            self.discrete_states = False

        if isinstance(env.action_space, spaces.Discrete):
            self.num_actions = env.action_space.n
            self.action_shape = []
            self.discrete_actions = True
        else:
            self.action_shape = env.action_space.shape
            self.discrete_actions = False


    def _build_output(self, out, action):
        if self.discrete_actions:
            self.out = tf.layers.dense(out,
                                       self.num_actions,
                                       name="logits",
                                       use_bias=False,
                                       activation=tf.nn.softmax)
            predict = tf.argmax(self.out, axis=1)

#             there is some bug with tf.multinomial
            stoch_predict = tf.multinomial(out, 1)[:, 0]

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.out,
                labels=action) * self.reward_placeholder
        else:
            out = tf.layers.dense(out,
                                  2 * np.prod(self.action_shape),
                                  name="actions",
                                  use_bias=False,
                                  activation=None)
            out = tf.reshape(out, [-1, 2] + list(self.action_shape))
            mu, sigma = out[:, 0], out[:, 1]
            eps = tf.random_normal(tf.shape(mu))

            predict = mu
            stoch_predict = mu + sigma * eps

            loss = tf.squared_difference(mu, action)

        return predict, stoch_predict, tf.reduce_mean(loss)


    def build(self):
        '''
            This function builds TF graph and all the ops
            belonging to it. As a result new members are acquired:
                - self.out: tensor [batch_size, action_shape]
                    action or their logits
                - self.predict: predicted action for greedy policy
                - self.update: train_op -- updates neural network using REINFORCE
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            state_shape = [None] + list(self.state_shape)
            action_shape = [None] + list(self.action_shape)

            self.state_placeholder = _placeholder(state_shape, name='state')
            self.reward_placeholder = _placeholder([None], name='reward')
            self.action_placeholder = _placeholder(action_shape, name='action', dtype=tf.int32)

            self.out = fc_network(self.state_placeholder,
                                  self.num_layers,
                                  self.num_units)

            self.predict, self.stoch_predict, self.loss = self._build_output(
                self.out,
                self.action_placeholder)

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
        if self.discrete_states:
            observation = _one_hot(observation, self.num_states)

        return observation



    def observe(self, old_observation, action, new_observation, reward, done):
        self.memory.insert(
            self.preprocess_state(old_observation),
            action,
            self.preprocess_state(new_observation),
            reward,
            done)

        if done and (self.episode_num + 1) % self.update_frequency == 0:
            discounted_rewards = self.memory.compute_returns(self.gamma)

            self.sess.run(self.update,
                          feed_dict={
                              self.state_placeholder: self.memory.old_states,
                              self.action_placeholder: self.memory.actions,
                              self.reward_placeholder: discounted_rewards
                              })


    def act(self, observation):
        feed_dict = {
            self.state_placeholder: [self.preprocess_state(observation)]
        }
        action_dist = self.sess.run(self.out, feed_dict)
        action = np.random.choice(self.num_actions, p=action_dist[0])
        return action


    def episode_end(self):
        if (self.episode_num + 1) % self.update_frequency == 0:
            self.memory.clear()

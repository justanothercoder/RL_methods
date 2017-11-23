# -*- coding: utf-8 -*-
'''
    This module defines QTable agent which uses Q-learning with tables.
'''

from agent import Agent

import numpy as np


class QTable(Agent):
    '''
        This class defines an Agent which uses Q-learning with
        state-action table.
        Params of __init__:
            - env: Environment -- environment to use;
            - gamma: float -- discount factor;
            - alpha: float -- learning rate.
    '''

    def __init__(self, env, gamma=0.95, alpha=0.8):
        super(QTable, self).__init__(env)

        n_states = env.observation_space.n
        n_actions = env.action_space.n

        self.Q_table = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.alpha = alpha


    def observe(self, old_observation, action, new_observation, reward):
        Q_new = reward + self.gamma * np.max(self.Q_table[new_observation, :])
        Q_old = self.Q_table[old_observation, action]

        self.Q_table[old_observation, action] = Q_old + self.alpha * (Q_new - Q_old)


    def act(self, observation):
        self.step_num += 1

        noise = np.random.randn(1, self.env.action_space.n) * (1. / (self.episode_num + 1))
        action = np.argmax(self.Q_table[observation, :] + noise)
        return action

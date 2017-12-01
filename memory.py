# -*- coding: utf-8 -*-

'''
    This module defines Memory class. It represents
    simple memory for one episode which contains
    transitions and can count auxiliary values such
    as returns or advantages.
'''

import numpy as np


def _discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0

    for t in reversed(range(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add

    return discounted_rewards


class Memory:
    '''
        This class defines simple episodic memory.
    '''

    def __init__(self):
        self.clear()


    def insert(self, old_state, action, new_state, reward, done):
        '''
            This method stores transitions.
            Params:
                - old_state: State -- old state of transition
                - action: Action -- action taken from old state
                - new_state: State -- new state of transition
                - reward: float -- reward r(old_state, action)
                - done: bool -- whether transition was last in episode
        '''
        self._old_states.append(old_state)
        self._new_states.append(new_state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._done.append(done)


    def compute_returns(self, gamma):
        '''
            This method computes and stores returns.
            Params:
                - gamma: float -- discount rate
            Returns:
                - discounted_rewards: np.array -- discounted rewards
        '''
        self._discounted_rewards = np.zeros_like(self.rewards)
        
        length = 0
        
        for t, d in enumerate(self._done):
            length += 1
        
            if d:
                self._discounted_rewards[t - length + 1: t + 1] = _discount_rewards(self.rewards[t - length + 1: t + 1], gamma)
                length = 0
                
#                print("DONE")
            
#        self._discounted_rewards = _discount_rewards(self.rewards, gamma)
        return self.discounted_rewards


    def clear(self):
        '''This method clears the memory.'''
        self._old_states = []
        self._new_states = []
        self._actions = []
        self._rewards = []
        self._done = []


    @property
    def old_states(self):
        '''Property: returns old states as array'''
        return np.array(self._old_states)


    @property
    def new_states(self):
        '''Property: returns new states as array'''
        return np.array(self._new_states)


    @property
    def actions(self):
        '''Property: returns actions as array'''
        return np.array(self._actions)


    @property
    def rewards(self):
        '''Property: returns rewards as array'''
        return np.array(self._rewards)


    @property
    def done(self):
        '''Property: returns done as array'''
        return np.array(self._done)


    @property
    def discounted_rewards(self):
        '''Property: returns discounted rewards as array'''
        return np.array(self._discounted_rewards)

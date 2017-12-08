# -*- coding: utf-8 -*-

'''
    This module defines Memory class. It represents
    simple memory for one episode which contains
    transitions and can count auxiliary values such
    as returns or advantages.
'''

import numpy as np


def _compute_advantages(rewards, values, done, gamma, lambd, next_pred):
    advantages = np.zeros_like(rewards)
    running_add = 0
     
    values = np.concatenate([values, next_pred[None]], axis=0)
    masks = np.concatenate([1 - done, np.ones([1, done.shape[1]])])
    
    for t in reversed(range(0, rewards.shape[0])):
        r = rewards[t]
        V_new = values[t + 1]
        V_old = values[t]
        mask = masks[t + 1]
        
        delta = r + gamma * V_new * mask - V_old
        advantages[t] = running_add = delta + gamma * lambd * mask * running_add
        
    returns = advantages + values[:-1]
    return advantages, returns


class Memory:
    '''
        This class defines simple episodic memory.
    '''

    def __init__(self):
        self.clear()


    def insert(self, old_state, action, new_state, reward, done, value=None):
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
        
        self._values.append(value)


    def compute_returns(self, gamma):
        '''
            This method computes and stores returns.
            Params:
                - gamma: float -- discount rate
            Returns:
                - discounted_rewards: np.array -- discounted rewards
        '''

        N, D = self.rewards.shape 
        
        self._discounted_rewards = np.zeros([N + 1, D])    

        self._discounted_rewards[-1] = 0
        masks = np.concatenate([1 - self.done, np.ones([1, 1])])
        
        for step in reversed(range(N)):
            R = self._discounted_rewards[step + 1]
            r = self.rewards[step]
            self._discounted_rewards[step] = r + R * gamma * masks[step + 1]

        self._discounted_rewards = self.discounted_rewards[:-1]
        return self.discounted_rewards


    def compute_advantages(self, gamma, lambd, next_pred):
        self._advantages = np.zeros_like(self.rewards)
        self._returns = np.zeros_like(self.rewards)
        
        adv, rets = _compute_advantages(
                self.rewards, self.values, self.done,
                gamma, lambd, next_pred)
        
        self._advantages = adv
        self._discounted_rewards = rets
                
        return self.advantages, self.discounted_rewards


    def clear(self):
        '''This method clears the memory.'''
        self._old_states = []
        self._new_states = []
        self._actions = []
        self._rewards = []
        self._done = []
        
        self._discounted_rewards = []
        self._values = []
        self._advantages = []


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


    @property
    def values(self):
        return np.array(self._values)


    @property
    def advantages(self):
        return np.array(self._advantages)
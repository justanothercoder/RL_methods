# -*- coding: utf-8 -*-

import numpy as np


def _discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    
    for t in reversed(range(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    
    return discounted_rewards


class Memory:
    def __init__(self):
        self.clear()
        
    
    def insert(self, old_state, action, new_state, reward, done):
        self._old_states.append(old_state)
        self._new_states.append(new_state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._done.append(done)


    def compute_returns(self, gamma):
        self._discounted_rewards = _discount_rewards(self.rewards, gamma)
        return self.discounted_rewards
    

    def clear(self):
        self._old_states = []
        self._new_states = []
        self._actions = []
        self._rewards = []
        self._done = []
        
    
    @property
    def old_states(self):
        return np.array(self._old_states)

    
    @property
    def new_states(self):
        return np.array(self._new_states)
    
    
    @property
    def actions(self):
        return np.array(self._actions)
    
    
    @property
    def rewards(self):
        return np.array(self._rewards)
    
    
    @property
    def done(self):
        return np.array(self._done)
    
    
    @property
    def discounted_rewards(self):
        return np.array(self._discounted_rewards)
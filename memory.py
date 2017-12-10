# -*- coding: utf-8 -*-

'''
    This module defines Memory class. It represents
    simple memory for one episode which contains
    transitions and can count auxiliary values such
    as returns or advantages.
'''

import numpy as np
from collections import namedtuple


Batch = namedtuple('Batch',
                   ['old_states',
                    'new_states',
                    'actions',
                    'rewards',
                    'done',
                    'returns',
                    'advantages',
                    'values',
                    'old_action_log_probability'])


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


    def insert(self, old_state, action, new_state, reward, done, value=None, action_logprob=None):
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
        self._old_action_log_probability.append(action_logprob)


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
        
        self._old_action_log_probability = []


    def batch_generator(self, batch_size, shuffle=True):
        N = len(self.old_states)
        
        indices = np.arange(N)
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, N - batch_size + 1, batch_size):
            ind = indices[i: i + batch_size]
            
            batch = Batch(
                    old_states=self.old_states[ind],
                    new_states=self.new_states[ind],
                    actions=self.actions[ind],
                    rewards=self.rewards[ind],
                    done=self.done[ind],
                    returns=self.discounted_rewards[ind],
                    advantages=self.advantages[ind],
                    values=self.values[ind],
                    old_action_log_probability=self.old_action_log_probability[ind]
                    )
            yield batch
            
        if N % batch_size != 0:
            ind = indices[i:]
            batch = Batch(
                    old_states=self.old_states[ind],
                    new_states=self.new_states[ind],
                    actions=self.actions[ind],
                    rewards=self.rewards[ind],
                    done=self.done[ind],
                    returns=self.discounted_rewards[ind],
                    advantages=self.advantages[ind],
                    values=self.values[ind],
                    old_action_log_probability=self.old_action_log_probability[ind]
                    )
            yield batch 


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
    
    
    @property
    def old_action_log_probability(self):
        return np.array(self._old_action_log_probability)
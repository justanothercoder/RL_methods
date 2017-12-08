# -*- coding: utf-8 -*-

'''
    This module defines an abstract class for all policies.
    Policy should be able to sample from it and return
    log probability of action.
'''


class Policy:
    '''Abstract class for all policies.'''

    @property
    def predict(self):
        '''This method should return policy greedy prediction.'''
        raise NotImplementedError


    @property
    def sample(self):
        '''This method should return sample from action distribution'''
        raise NotImplementedError


    @property
    def log_probability(self):
        '''This method should return log probability of given action'''
        raise NotImplementedError


    @property
    def value_pred(self):
        '''This method should return prediction of value function'''
        raise NotImplementedError
    

    @property
    def state(self):
        '''This method should return state placeholder'''
        raise NotImplementedError


    @property
    def action(self):
        '''This method should return action placeholder'''
        raise NotImplementedError

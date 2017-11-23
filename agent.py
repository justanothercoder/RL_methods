# -*- coding: utf-8 -*-
'''
    This module describes an abstract class for all agents.
'''

class Agent:
    '''
        Abstract class for an RL agent.
        Methods to override:
            - observe
            - act
    '''
    def __init__(self, env):
        self.env = env
        self.episode_num = 0
        self.step_num = None


    def observe(self, old_observation, action, new_observation, reward):
        '''
            This method should be implemented in subclasses.
            It should handle observation of transition of agent in
            an environment.
            Params:
                - old_observation -- old state of environment;
                - action -- an action which agent took last time;
                - new_observation -- new state of environment;
                - reward -- a reward which agent received last time.
        '''
        raise NotImplementedError


    def act(self, observation):
        '''
            This method should be implemented in subclasses.
            Params:
                - observation -- observed state of environment.
        '''
        raise NotImplementedError


    def episode_start(self):
        '''
            Helper method to count number of episodes agent
            took.
        '''
        self.step_num = 0
        
        
    def episode_end(self):
        '''
        '''
        self.episode_num += 1
        
    
    def step_start(self):
        '''
        '''
        pass
    
    
    def step_end(self):
        self.step_num += 1

# -*- coding: utf-8 -*-
'''
    This module defines Trainer class which is used
    to encapsulate training of reinforcement learning
    agent in an environment.
'''

import numpy as np


class Trainer:
    '''
        This class defines some 'train' method which
        can be used to train an RL agent.
        Params of __init__:
            - env: Environment -- environment to use;
            - agent: Agent -- agent to train;
            - config: Config -- configuration dict to use.
    '''
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config


    def train(self, num_episodes, max_steps):
        '''
            This function trains agent to act in environment.
            Params:
                - num_episodes: int -- number of episodes to run;
                - max_steps: int -- maximum number of steps during one episode.
        '''
        rewards = []
        lengths = []

        for i in range(num_episodes):
            episode_reward, episode_length = self.run_episode(max_steps)
            
            rewards.append(episode_reward)
            lengths.append(episode_length)

            print("\rMean reward: {:5f}".format(np.mean(rewards)), end='')
#            print("Mean reward: {:5f}".format(np.mean(rewards)))
            
        return rewards, lengths


    def run_episode(self, max_steps):
        '''
            This function runs training of agent on one episode.
            Params:
                - max_steps: int -- maximum number of steps during one episode.
        '''
        state = self.env.reset()
        done = False
        episode_reward = 0.
        step_num = 0

        self.agent.episode_start()

        while not done and step_num < max_steps:
            self.agent.step_start()
            
            step_num += 1
            action = self.agent.act(state)

            new_state, reward, done, _ = self.env.step(action)
            self.agent.observe(state, action, new_state, reward)

            episode_reward += reward
            state = new_state
            
            self.agent.step_end()
            
        self.agent.episode_end()
            
        return episode_reward, step_num

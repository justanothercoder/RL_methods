# -*- coding: utf-8 -*-
'''
    This module defines Trainer class which is used
    to encapsulate training of reinforcement learning
    agent in an environment.
'''

from trainer import Trainer

import gym
import numpy as np

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


class ActorCriticTrainer(Trainer):
    '''
        This class defines some 'train' method which
        can be used to train an RL agent.
        Params of __init__:
            - env: Environment -- environment to use;
            - agent: Agent -- agent to train;
            - config: Config -- configuration dict to use.
    '''
    def __init__(self, env, agent, config, log_dir='logs'):
        super(ActorCriticTrainer, self).__init__(env, agent, config)        
        
        def create_env(seed):
            _env = self.env
            def _thunk():
                name = _env.unwrapped.spec.id
                env = gym.make(name)
                env.seed(seed)
                
#                if log_dir is not None:
#                    env = bench.Monitor(env, os.path.join(log_dir, str(seed)))

                return env
            
            return _thunk
        
        envs = [
            create_env(i)
            for i in range(self.config.num_processes)
        ]
        
        if self.config.num_processes == 1:
            self.env = DummyVecEnv(envs)
        else:
            self.env = SubprocVecEnv(envs)


    def run_episode(self, max_steps):
        '''
            This function runs training of agent on one episode.
            Params:
                - max_steps: int -- maximum number of steps during one episode.
        '''
        state = self.env.reset()
        episode_reward = np.zeros(self.config.num_processes)
        final_reward = np.zeros(self.config.num_processes)
        
        self.agent.episode_start()

        for step_num in range(max_steps):
#            self.env.render()
            self.agent.step_start()
            
            step_num += 1
            action, value_pred = self.agent.act(state)

            new_state, reward, done, _ = self.env.step(action)
            self.agent.observe(state, action, new_state, reward, done, value_pred)

            masks = 1 - done
            episode_reward += reward
            
            final_reward *= masks
            final_reward += episode_reward * (1 - masks)
            
            episode_reward *= masks
            
            state = new_state
            self.agent.step_end()
                
        self.agent.episode_end()
            
        final_reward = np.mean(final_reward)
        step_num = np.mean(1 - done)
        return final_reward, step_num
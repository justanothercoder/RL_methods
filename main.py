# -*- coding: utf-8 -*-
'''
Main module
'''

import gym

from config import Config
from qtable import QTable
from trainer import Trainer


def main(flags):
    '''
        Runs an agent in an environment.
        params:
            flags (dict): configuration
    '''
    env = gym.make('FrozenLake-v0')
    agent = QTable(env, gamma=flags.gamma, alpha=flags.alpha)

    trainer = Trainer(env, agent, flags)
    trainer.train(flags.num_episodes, flags.max_steps)

    print("Final Q-table: {}".format(agent.Q_table))


if __name__ == '__main__':
    main(Config(
        num_episodes=2000,
        alpha=0.8,
        gamma=0.95,
        max_steps=100,
    ))

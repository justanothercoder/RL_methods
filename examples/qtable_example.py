# -*- coding: utf-8 -*-

'''
    This file shows an example of using QTable agent for 
    FrozenLake-v0 environment.
'''

import gym
import matplotlib.pyplot as plt

from config import Config
from qtable import QTable

from trainer import Trainer


def plot_results(rewards, lengths):
    '''
        Simple function to plot rewards over time and
        change of typical episode length
    '''
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    axes[0].plot(rewards)
    axes[1].plot(lengths)

    return fig


def main(flags):
    '''
        Runs an agent in an environment.
        params:
            flags (dict): configuration
    '''
    env = gym.make('FrozenLake-v0')
    agent = QTable(env,
                   gamma=flags.gamma,
                   alpha=flags.learning_rate)

    trainer = Trainer(env, agent, flags)
    rewards, lengths = trainer.train(flags.num_episodes, flags.max_steps)

    plot_results(rewards, lengths)


if __name__ == '__main__':
    table_config = Config(
        num_episodes=2000,
        learning_rate=0.8,
        gamma=0.95,
        max_steps=100,
    )

    main(table_config)

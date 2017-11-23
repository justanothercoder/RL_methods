# -*- coding: utf-8 -*-

'''
    This file shows an example of using Q-learning with
    neural networks on FrozenLake-v0 Gym environment.
'''

import gym
import matplotlib.pyplot as plt

from config import Config
from qnetwork import QNetwork

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

    agent = QNetwork(env,
                     gamma=flags.gamma,
                     learning_rate=flags.learning_rate,
                     num_units=flags.num_units,
                     num_layers=flags.num_layers)

    trainer = Trainer(env, agent, flags)
    rewards, lengths = trainer.train(flags.num_episodes, flags.max_steps)

    plot_results(rewards, lengths)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    network_config = Config(
        num_episodes=2000,
        learning_rate=0.1,
        gamma=0.99,
        max_steps=100,
        num_units=100,
        num_layers=0,
    )

    main(network_config)

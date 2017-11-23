# -*- coding: utf-8 -*-
'''
Main module
'''

import gym
import matplotlib.pyplot as plt

from config import Config

from qtable import QTable
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
    env.seed(42)

    import numpy as np
    np.random.seed(42)

    import tensorflow as tf
    tf.set_random_seed(42)

    agent = QNetwork(env,
                     gamma=flags.gamma,
                     learning_rate=flags.learning_rate,
                     num_units=flags.num_units,
                     num_layers=flags.num_layers)
#    agent = QTable(env,
#                   gamma=flags.gamma,
#                   alpha=flags.learning_rate)

    trainer = Trainer(env, agent, flags)
    rewards, lengths = trainer.train(flags.num_episodes, flags.max_steps)

    plot_results(rewards, lengths)

#    print("Final Q-table: {}".format(agent.Q_table))


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    table_config = Config(
        num_episodes=2000,
        learning_rate=0.8,
        gamma=0.95,
        max_steps=100,
    )

    network_config = Config(
        num_episodes=2000,
        learning_rate=0.1,
        gamma=0.99,
        max_steps=100,
        num_units=100,
        num_layers=0,
    )

    main(network_config)

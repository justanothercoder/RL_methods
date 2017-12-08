# -*- coding: utf-8 -*-

'''
    This file shows an example of using REINFORCE algorithm with
    neural networks on CartPole-v0 Gym environment.
'''

import gym
import matplotlib.pyplot as plt

from config import Config
from a2c import A2C

from actor_critic_trainer import ActorCriticTrainer


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
    env = gym.make('CartPole-v0')

    agent = A2C(env,
                      gamma=flags.gamma,
                      lambd=flags.lambd,
                      learning_rate=flags.learning_rate,
                      num_units=flags.num_units,
                      num_layers=flags.num_layers,
                      update_frequency=flags.update_frequency)

    trainer = ActorCriticTrainer(env, agent, flags)
    rewards, lengths = trainer.train(flags.num_episodes, flags.max_steps)

    plot_results(rewards, lengths)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    a2c_config = Config(
        num_episodes=4000,
        learning_rate=0.001,
        gamma=1.,
        lambd=0.7,
        max_steps=1000,
        num_units=200,
        num_layers=1,
        update_frequency=1,
        num_processes=1 #3
    )

    main(a2c_config)

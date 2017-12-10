# -*- coding: utf-8 -*-

'''
    This file shows an example of using REINFORCE algorithm with
    neural networks on CartPole-v0 Gym environment.
'''

import gym
import matplotlib.pyplot as plt

from config import Config
from ppo import PPO

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

    agent = PPO(env,
                gamma=flags.gamma,
                lambd=flags.lambd,
                learning_rate=flags.learning_rate,
                num_units=flags.num_units,
                num_layers=flags.num_layers,
                update_frequency=flags.update_frequency,
                clip_param=flags.clip_param,
                ppo_epochs=flags.ppo_epochs,
                normalize_advantages=flags.normalize_advantages,
                batch_size=flags.batch_size)

    trainer = ActorCriticTrainer(env, agent, flags)
    rewards, lengths = trainer.train(flags.num_episodes, flags.max_steps)

    plot_results(rewards, lengths)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
   
    a2c_config = Config(
        num_episodes=4000,
        learning_rate=0.01,
        gamma=1.,
        lambd=0.99,
        max_steps=1000,
        num_units=200,
        num_layers=1,
        update_frequency=1,
        num_processes=4, #3
        clip_param=0.1,
        normalize_advantages=True,
        ppo_epochs=5,
        batch_size=64
    )

    main(a2c_config)

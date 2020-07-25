import argparse
from collections import defaultdict
import random

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import RescaleAction
from tqdm import tqdm

from mbpo.env_wrappers import ActionRepeat, ObservationNormalize, TestObservationNormalize
from mbpo.mbpo import MBPO
from mbpo.utils import TrainingLogger


def define_config():
    return {
        # MBPO
        'horizon': 5,
        'update_steps': 1,
        'discount': 0.99,
        'lambda_': 0.95,
        'steps_per_update': 1000,
        'model_rollouts': 400,
        'warmup_training_steps': 5000,
        # MODELS
        'dynamics_layers': 4,
        'units': 128,
        'ensemble_size': 5,
        'model_learning_rate': 2.5e-4,
        'actor_learning_rate': 3e-5,
        'critic_learning_rate': 3e-5,
        'clip_norm': 5.0,
        'weight_decay': 1e-5,
        'critic_regularization': 1e-3,
        # TRAINING
        'total_training_steps': 100000,
        'action_repeat': 3,
        'environment': 'InvertedPendulum-v2',
        'seed': 314,
        'steps_per_log': 1000,
        'episode_length': 1000,
        'training_steps_per_epoch': 10000,
        'evaluation_steps_per_epoch': 5000,
        'log_dir': None,
        'render_episodes': 1
    }


def do_episode(agent, training, environment, config, pbar, render):
    observation = environment.reset()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation, training)
        next_observation, reward, terminal, info = environment.step(action)
        agent.observe(dict(observation=observation,
                           next_observation=next_observation,
                           action=action,
                           reward=reward,
                           terminal=terminal,
                           info=info))
        if render:
            episode_summary['image'].append(environment.render(mode='rgb_array'))
        pbar.update(config.action_repeat)
        steps += config.action_repeat
        done = terminal or steps >= config.episode_length
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
        observation = next_observation
    return steps, episode_summary


def interact(agent, environment, steps, config, training=True):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []
    while steps_count < steps:
        episode_steps, episode_summary = \
            do_episode(agent, training,
                       environment, config,
                       pbar, len(episodes) < config.render_episodes and not training)
        steps_count += episode_steps
        episodes.append(episode_summary)
    pbar.close()
    return steps, episodes


def make_env(name, action_repeat):
    env = gym.make(name)
    env = ActionRepeat(env, action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    train_env = ObservationNormalize(env)
    test_env = TestObservationNormalize(train_env)
    return train_env, test_env


def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    logger = TrainingLogger(config.log_dir)
    train_env, test_env = make_env(config.environment, config.action_repeat)
    agent = MBPO(config, logger, train_env.observation_space, train_env.action_space)
    steps = 0
    while steps < config.total_training_steps:
        print("Training epoch.")
        training_steps, training_episodes_summaries = interact(
            agent, train_env, config.training_steps_per_epoch, config, training=True)
        steps += training_steps
        print("Evaluating.")
        evaluation_steps, evaluation_episodes_summaries = interact(
            agent, test_env, config.evaluation_steps_per_epoch, config, training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument('--{}'.format(key), type=type(value), default=value)
    main(parser.parse_args())

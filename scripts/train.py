import argparse
import os
import random

import numpy as np
import tensorflow as tf

import mbpo.utils as utils
from mbpo.mbpo import MBPO


def define_config():
    return {
        # MBPO
        'horizon': 5,
        'update_steps': 100,
        'discount': 0.99,
        'lambda_': 0.95,
        'steps_per_update': 1000,
        # This batch size is split evenly among world models in the ensemble but gets as a whole
        # for the actor critic. This *might* reduce the high variance of the actor-critic and
        # still be not too large batch size for the model.
        'batch_size': 320,
        'warmup_training_steps': 5000,
        # MODELS
        'dynamics_layers': 4,
        'units': 128,
        'ensemble_size': 5,
        'model_learning_rate': 2.5e-4,
        'actor_learning_rate': 3e-5,
        'critic_learning_rate': 3e-5,
        'grad_clip_norm': 5.0,
        'weight_decay': 1e-5,
        'critic_regularization': 1e-3,
        # TRAINING
        'total_training_steps': 100000,
        'action_repeat': 3,
        'filter_goal_mets': False,
        'environment': 'InvertedPendulum-v2',
        'seed': 314,
        'steps_per_log': 1000,
        'episode_length': 1000,
        'training_steps_per_epoch': 10000,
        'evaluation_steps_per_epoch': 5000,
        'log_dir': 'runs',
        'render_episodes': 1,
        'debug_model': False,
        'cuda_device': '-1'
    }


def main(config):
    logger = utils.TrainingLogger(config)
    random.seed(config.seed)
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)
    train_env, test_env = utils.make_env(config.environment, config.episode_length,
                                         config.action_repeat, config.seed)
    agent = MBPO(config, logger, train_env.observation_space, train_env.action_space)
    steps = 0
    while steps < config.total_training_steps:
        print("Performing a training epoch.")
        training_steps, training_episodes_summaries = utils.interact(
            agent, train_env, config.training_steps_per_epoch, config, training=True)
        steps += training_steps
        print("Evaluating.")
        evaluation_steps, evaluation_episodes_summaries = utils.interact(
            agent, test_env, config.evaluation_steps_per_epoch, config, training=False)
        eval_summary = dict(eval_score=np.asarray([
            sum(episode['reward']) for episode in evaluation_episodes_summaries]).mean(),
                            episode_length=np.asarray([
                                episode['steps'][0]
                                for episode in evaluation_episodes_summaries]).mean(),
                            training_sum_rewards=np.asarray([
                                sum(episode['reward']) for episode in training_episodes_summaries
                            ]).mean())
        if config.render_episodes and config.evaluation_steps_per_epoch:
            video = evaluation_episodes_summaries[config.render_episodes - 1].get('image')
            logger.log_video(video, steps)
        if config.debug_model:
            eval_summary.update(utils.evaluate_model(evaluation_episodes_summaries, agent))
        logger.log_evaluation_summary(eval_summary, steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument('--{}'.format(key), type=type(value) if value else str, default=value)
    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_device
    main(config)

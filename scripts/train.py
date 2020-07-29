import argparse
import random
import numpy as np
import tensorflow as tf


from mbpo.mbpo import MBPO
import mbpo.utils as utils


def define_config():
    return {
        # MBPO
        'horizon': 5,
        'discount': 0.99,
        'steps_per_model_update': 1000,
        'model_rollouts': 400,
        'warmup_training_steps': 5000,
        'model_grad_steps': 500,
        'actor_critic_update_steps': 25,
        'model_batch_size': 128,
        'actor_critic_batch_size': 512,
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
        'log_dir': None,
        'render_episodes': 1
    }


def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    logger = utils.TrainingLogger(config.log_dir)
    train_env, test_env = utils.make_env(config.environment, config.action_repeat)
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
            sum(episode['reward'])
            for episode in evaluation_episodes_summaries]).mean())
        logger.log_evaluation_summary(eval_summary, training_steps)
        for episode in evaluation_episodes_summaries:
            video = episode.get('image', None)
            if video:
                logger.log_video(video, training_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument('--{}'.format(key), type=type(value), default=value)
    main(parser.parse_args())

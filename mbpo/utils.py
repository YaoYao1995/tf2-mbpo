import random
from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym.wrappers import RescaleAction
from tensorboardX import SummaryWriter
from tqdm import tqdm

from mbpo.env_wrappers import ActionRepeat, ObservationNormalize, TestObservationNormalize


# Following https://github.com/tensorflow/probability/issues/840 and
# https://github.com/tensorflow/probability/issues/840.
class StableTanhBijector(tfp.bijectors.Tanh):
    def __init__(self, validate_args=False, name='tanh_stable_bijector'):
        super(StableTanhBijector, self).__init__(validate_args=validate_args, name=name)

    def _inverse(self, y):
        y = tf.where(
            tf.less_equal(tf.abs(y), 1.),
            tf.clip_by_value(y, -1.0 + 1e-6, 1.0 - 1e-6),
            y)
        return tf.atanh(y)


class SampleDist(object):
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)


class TrainingLogger(object):
    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir)
        self._metrics = defaultdict(tf.metrics.Mean)

    def __getitem__(self, item):
        return self._metrics[item]

    def __setitem__(self, key, value):
        self._metrics[key] = value

    def log_evaluation_summary(self, summary, step):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)
        self._writer.flush()

    def log_metrics(self, step):
        print("Training step {} summary:".format(step))
        for k, v in self._metrics.items():
            print("{:<40} {:<.2f}".format(k, float(v.result())))
            self._writer.add_scalar(k, float(v.result()), step)
            v.reset_states()
        self._writer.flush()

    def log_video(self, images, step):
        video = np.expand_dims(np.transpose(images, [0, 3, 1, 2]), axis=0)
        self._writer.add_video('Evaluation policy', video, step, fps=15)
        self._writer.flush()


def do_episode(agent, training, environment, config, pbar, render):
    observation = environment.reset()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation, training).squeeze()
        next_observation, reward, done, info = environment.step(action)
        terminal = done and not info.get('TimeLimit.truncated')
        if training:
            agent.observe(dict(observation=observation.astype(np.float32),
                               next_observation=next_observation.astype(np.float32),
                               action=action.astype(np.float32),
                               reward=np.array([reward], dtype=np.float32),
                               terminal=np.array([terminal], dtype=np.bool),
                               info=np.array([info], dtype=dict),
                               steps=info.get('steps', config.action_repeat)))
        observation = next_observation
        if render:
            episode_summary['image'].append(environment.render(mode='rgb_array'))
        pbar.update(info.get('steps', config.action_repeat))
        steps += info.get('steps', config.action_repeat)
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['action'].append(action)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
    episode_summary['steps'] = [steps]
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


def make_env(name, episode_length, action_repeat):
    env = gym.make(name)
    if not isinstance(env, gym.wrappers.TimeLimit):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    else:
        # https://github.com/openai/gym/issues/499
        env._max_episode_steps = episode_length
    env = ActionRepeat(env, action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    train_env = ObservationNormalize(env)
    test_env = TestObservationNormalize(env, train_env.normalize)
    return train_env, test_env


# Reading the errors produced by this function should assume all obsersvations are normalized to
# [-1, 1]
def debug_model(episodes_summaries, agent):
    observations_mse = 0.0
    rewards_mse = 0.0
    terminal_accuracy = 0.0
    n_episodes = min(len(episodes_summaries), 30)
    for i in range(n_episodes):
        observations = tf.expand_dims(
            tf.constant(episodes_summaries[i]['observation'][0], dtype=tf.float32), axis=0)
        actions = tf.constant(episodes_summaries[i]['action'], dtype=tf.float32)
        actions = actions[:, tf.newaxis, tf.newaxis] if tf.rank(actions) < 2 else actions
        predicted_rollouts = agent.imagine_rollouts(observations, random.choice(agent.ensemble),
                                                    actions)
        observations_mse += (np.asarray(predicted_rollouts['next_observation'].numpy()
                                        - episodes_summaries[i][
                                            'next_observation']) ** 2).mean() / n_episodes
        rewards_mse += (np.asarray(predicted_rollouts['reward'].numpy() -
                                   episodes_summaries[i]['reward']) ** 2).mean() / n_episodes
        terminal_accuracy += (np.abs(predicted_rollouts['terminal'] -
                                     episodes_summaries[i][
                                         'terminal']) < 1e-5).all().mean() / n_episodes
    return dict(observations_mse=observations_mse,
                rewards_mse=rewards_mse,
                terminal_accuracy=terminal_accuracy)

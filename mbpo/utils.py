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
        self._writer.add_video('Evaluation policy', video, step)
        self._writer.flush()


def do_episode(agent, training, environment, config, pbar, render):
    observation = environment.reset()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation, training)
        next_observation, reward, terminal, info = environment.step(action)
        agent.observe(dict(observation=observation.astype(np.float32),
                           next_observation=next_observation.astype(np.float32),
                           action=action.astype(np.float32),
                           reward=np.array([reward], dtype=np.float32),
                           terminal=np.array([terminal], dtype=np.bool),
                           info=np.array([info], dtype=dict)))
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


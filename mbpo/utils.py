from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorboardX import SummaryWriter


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
        video = np.transpose(images, [0, 3, 1, 2])
        self._writer.add_video('Evaluation policy', video, step)
        self._writer.flush()

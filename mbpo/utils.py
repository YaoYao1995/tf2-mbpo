from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def gaussian_negative_log_likelihood(y_true, mu, var, squashed=False):
    likelihood = 0.5 * tf.reduce_sum(tf.math.log(2.0 * np.pi * var), axis=1) + \
                 0.5 * tf.reduce_sum(tf.math.divide(tf.square(mu - y_true), var), axis=1)
    if squashed:
        return likelihood - tf.reduce_sum(tf.math.log(1 - tf.tanh(mu) ** 2 + 1e-6), axis=1)
    else:
        return likelihood


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


class TrainingLogger(object):
    def __init__(self, writer):
        self._writer = writer
        self._metrics = defaultdict(tf.metrics.Mean)

    def __getitem__(self, item):
        return self._metrics[item]

    def __setitem__(self, key, value):
        self._metrics[key] = value

    def log_metrics(self, step):
        print("Training step {} summary:".format(step))
        for k, v in self._metrics.items():
            print("{:<40} {:<.2f}".format(k, v))
            self._writer.add_scalar(k, float(v.result()), step)
            v.reset_states()
        self._writer.flush()

    def log_video(self):
        pass

import tensorflow as tf
from tensorflow_probability import distributions as tfd

import mbpo.utils as utils


class WorldModel(tf.Module):
    def __init__(self, dynamics_size, dynamics_layers, units, reward_layers=1, terminal_layers=1,
                 min_stddev=1e-4,
                 activation=tf.nn.relu):
        super().__init__()
        self._dynamics = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(dynamics_layers)]
        )
        self._next_observation_residual_mu = tf.keras.layers.Dense(dynamics_size)
        self._next_observation_stddev = tf.keras.layers.Dense(
            dynamics_size,
            activation=lambda t: tf.math.softplus(t) + min_stddev)
        # Assuming reward with a unit standard deviation.
        # TODO (yarden): not sure if this is too simplifying.
        self._reward_mu = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(reward_layers)] + [
                tf.keras.layers.Dense(1)])
        self._terminal_logit = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in
             range(terminal_layers)] + [
                tf.keras.layers.Dense(1)])

    def __call__(self, observation, action):
        cat = tf.concat([observation, action], axis=1)
        x = self._dynamics(cat)
        # TODO (yarden): maybe it's better to feed the reward and terminals s, a instead of x.
        # The world model predicts the difference between next_observation and observation.
        return dict(next_observation=tfd.MultivariateNormalDiag(
            loc=self._next_observation_residual_mu(x) + tf.stop_gradient(observation),
            scale_diag=self._next_observation_stddev(x)),
            reward=tfd.Normal(loc=self._reward_mu(x), scale=1.0),
            terminal=tfd.Bernoulli(logits=self._terminal_logit(x), dtype=tf.float32))


class Actor(tf.Module):
    def __init__(self, size, layers, units, min_stddev=1e-4, activation=tf.nn.relu):
        super().__init__()
        self._policy = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(layers)]
        )
        self._mu = tf.keras.layers.Dense(size)
        self._stddev = tf.keras.layers.Dense(
            size,
            activation=lambda t: tf.math.softplus(t) + min_stddev)

    def __call__(self, observation):
        x = self._policy(observation)
        multivariate_normal_diag = tfd.MultivariateNormalDiag(
            loc=self._mu(x),
            scale_diag=self._stddev(x))
        # Squash actions to [-1, 1]
        squashed = tfd.TransformedDistribution(multivariate_normal_diag, utils.StableTanhBijector())
        return utils.SampleDist(squashed)


class Critic(tf.Module):
    def __init__(self, layers, units, activation=tf.nn.relu, output_regularization=1e-3):
        super().__init__()
        self._action_value = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(layers)]
        )
        self._action_value.layers.append(
            tf.keras.layers.Dense(units=1, activity_regularizer=tf.keras.regularizers.l2(
                output_regularization)))

    def __call__(self, observation):
        mu = self._action_value(observation)
        return tfd.Normal(loc=mu, scale=1.0)

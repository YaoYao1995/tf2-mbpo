import tensorflow as tf


class WorldModel(tf.Module):
    def __init__(self, dynamics_size, dynamics_layers, units, reward_layers=1, terminal_layers=1, min_stddev=1e-4,
                 activation=tf.nn.relu):
        super().__init__()
        self._dynamics = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(dynamics_layers)]
        )
        self._next_state_mu = tf.keras.layers.Dense(dynamics_size)
        self._next_statet_var = tf.keras.layers.Dense(
            dynamics_size,
            activation=lambda t: tf.math.softplus(t) + min_stddev)
        # Assuming reward with a unit standard deviation.
        # TODO (yarden): not sure if this is too simplifying
        self._reward_mu = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(reward_layers)] + [
                tf.keras.layers.Dense(1)])
        self._terminal_logit = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(terminal_layers)] + [
                tf.keras.layers.Dense(1)])

    def __call__(self, state, action):
        x = self._dynamics(tf.concat([state, action], axis=1))
        return self._next_state_mu(x), self._next_statet_var(x), self._reward_mu(x), self._terminal_logit(x)


class Actor(tf.Module):
    def __init__(self, size, layers, units, min_stddev=1e-4, activation=tf.nn.relu):
        super().__init__()
        self._policy = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(layers)]
        )
        self._mu = tf.keras.layers.Dense(size)
        self._var = tf.keras.layers.Dense(
            size,
            activation=lambda t: tf.math.softplus(t) + min_stddev)

    def __ceil__(self, state):
        # TODO (yarden): think of squashing here as par with
        #  https://github.com/openai/safety-starter-agents/blob/master/safe_rl/sac/sac.py
        x = self._policy(state)
        # TODO (yarden): tanh?
        return self._mu(x), self._var(x)


class Critic(tf.Module):
    def __init__(self, layers, units, activation=tf.nn.relu):
        super().__init__()
        # TODO (yarden): verify that the fact that there is no last dense(1) layer is a bug.
        self._action_value = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(layers)]
        )

    def __call__(self, state, action):
        return self._action_value(tf.concat([state, action], axis=1))

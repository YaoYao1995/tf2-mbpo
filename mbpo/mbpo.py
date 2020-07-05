import tensorflow as tf

import mbpo.models as models
from mbpo.replay_buffer import ReplayBuffer


class MBPO(tf.Module):
    def __init__(self, config, writer, observation_space, action_space):
        super(MBPO, self).__init__()
        self._config = config
        self._writer = writer
        self._model_data = ReplayBuffer()
        self._environment_data = ReplayBuffer()
        self._build(observation_space, action_space)

    def _build(self, observation_space, action_space):
        self._ensemble = [models.WorldModel(
            observation_space.shape[0],
            self._config.dynamics_layers,
            self._config.units, reward_layers=2, terminal_layers=2) for _ in range(self._config.ensemble_size)]
        self._actor = models.Actor(action_space.shape[0], 3, self._config.units)
        self._critic_target = models.Critic(3, self._config.units)
        self._critic_king = models.Critic(3, self._config.units)
        self._critic_queen = models.Critic(3, self._config.units)

    def sample_rollouts(self, training=True):
        pass

    def imagine_rollouts(self):
        pass

    def update_model(self):
        pass

    def update_actor_critic(self):
        pass
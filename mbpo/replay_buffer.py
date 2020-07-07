import numpy as np


class ReplayBuffer(object):
    def __init__(self, observation_space_dim, action_space_dim, buffer_size=1000000):
        self._buffer_size = buffer_size
        self._observations = np.empty(shape=(0, observation_space_dim), dtype=np.float32)
        self._next_observations = np.empty(shape=(0, observation_space_dim), dtype=np.float32)
        self._actions = np.empty(shape=(0, action_space_dim), dtype=np.float32)
        self._rewards = np.empty(shape=(0,), dtype=np.float32)
        self._terminals = np.empty(shape=(0,), dtype=np.bool)
        self._infos = np.empty(shape=(0,), dtype=dict)

    def store(self, rollouts):
        self._observations = np.concatenate(
            self._observations, rollouts['observation'].flatten()[-self._buffer_size:]
        )
        self._next_observations = np.concatenate(
            self._next_observations, rollouts['next_observation'].flatten()[-self._buffer_size:]
        )
        self._actions = np.concatenate(
            self._actions, rollouts['action'].flatten()[-self._buffer_size:]
        )
        self._rewards = np.concatenate(
            self._rewards, rollouts['reward'].flatten()[-self._buffer_size:]
        )
        self._terminals = np.concatenate(
            self._terminals, rollouts['terminal'].flatten()[-self._buffer_size:]
        )
        self._infos = np.concatenate(
            self._infos, rollouts['info'].flatten()[-self._buffer_size:]
        )

    def sample(self, batch_size):
        indices = np.random.permutation(self._observations.shape[0])[-batch_size:]
        return self._observations[indices], \
               self._next_observations[indices], \
               self._actions[indices], \
               self._rewards[indices], \
               self._terminals[indices], \
               self._infos[indices]

import numpy as np


class ReplayBuffer(object):
    def __init__(self, observation_space_dim, action_space_dim, buffer_size=1000000):
        self._buffer_size = buffer_size
        self._observations = np.empty(shape=(buffer_size, observation_space_dim), dtype=np.float32)
        self._next_observations = np.empty(shape=(buffer_size, observation_space_dim),
                                           dtype=np.float32)
        self._actions = np.empty(shape=(buffer_size, action_space_dim), dtype=np.float32)
        self._rewards = np.empty(shape=(buffer_size,), dtype=np.float32)
        self._terminals = np.empty(shape=(buffer_size,), dtype=np.bool)
        self._infos = np.empty(shape=(buffer_size,), dtype=dict)
        self._ptr = 0

    def store(self, rollouts):
        length = len(rollouts['observation'].flatten())
        self._observations[self._ptr:self._ptr + length] = \
            np.asarray(rollouts['observation']).flatten()
        self._next_observations[self._ptr:self._ptr + length] = \
            np.asarray(rollouts['next_observation']).flatten()
        self._actions[self._ptr:self._ptr + length] = \
            np.asarray(rollouts['action']).flatten()
        self._rewards[self._ptr:self._ptr + length] = \
            np.asarray(rollouts['reward']).flatten()
        self._terminals[self._ptr:self._ptr + length] = \
            np.asarray(rollouts['terminal']).flatten()
        self._infos[self._ptr:self._ptr + length] = \
            np.asarray(rollouts['info']).flatten()
        self._ptr = (self._ptr + length) % self._buffer_size

    def sample(self, batch_size, filter_goal_mets=False):
        indices = np.random.permutation(self._ptr)[-batch_size:]
        observations, next_observations, actions, rewards, terminals, infos = \
            self._observations[indices], \
            self._next_observations[indices], \
            self._actions[indices], \
            self._rewards[indices], \
            self._terminals[indices], \
            self._infos[indices]
        if filter_goal_mets:
            goal_mets = np.array(list(map(lambda info: info.get('goal_met', False), infos)))
            # We mask transitions with 'goal_met' since they are non-continuous, what extremely
            # destabilizes the learning of p(s_t_1 | s_t, a_t)
            observations, actions, next_observations, rewards = \
                observations[~goal_mets, ...], actions[~goal_mets, ...], \
                next_observations[~goal_mets, ...], rewards[~goal_mets, ...]
        return {'observation': observations,
                'next_observation': next_observations,
                'action': actions,
                'reward': rewards,
                'terminal': terminals}

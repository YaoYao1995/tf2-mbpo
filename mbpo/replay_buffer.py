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

    def sample(self, batch_size, filter_goal_mets=False):
        indices = np.random.permutation(self._observations.shape[0])[-batch_size:]
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
                observations[~goal_mets, ...], actions[~goal_mets, ...], next_observations[
                    ~goal_mets, ...], rewards[~goal_mets, ...]
        return {'observation': observations,
                'next_observation': next_observations,
                'action': actions,
                'reward': rewards,
                'terminal': terminals}

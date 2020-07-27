import numpy as np


class ReplayBuffer(object):
    def __init__(self, observation_dim, action_dim, buffer_horizon, buffer_capacity=1000000):
        self._buffer_capacity = buffer_capacity
        self._buffers = {
            'observation': np.empty((buffer_capacity, buffer_horizon, observation_dim),
                                    dtype=np.float32),
            'next_observation': np.empty((buffer_capacity, buffer_horizon, observation_dim),
                                         dtype=np.float32),
            'action': np.empty((buffer_capacity, buffer_horizon, action_dim), dtype=np.float32),
            'reward': np.empty((buffer_capacity, buffer_horizon, 1), dtype=np.float32),
            'terminal': np.empty((buffer_capacity, buffer_horizon, 1), dtype=np.bool),
            'info': np.empty((buffer_capacity, buffer_horizon, 1), dtype=dict)
        }
        self._size = 0
        self._ptr = 0

    def store(self, rollouts):
        assert rollouts['observation'].shape[1] == self._buffers['observation'].shape[1]
        length = rollouts['observation'].shape[0]
        self._size += length
        for k, v in rollouts.items():
            self._buffers[k][self._ptr:self._ptr + length, ...] = rollouts[k]
        self._ptr = (self._ptr + length) % self._buffer_capacity

    def sample(self, batch_size, horizon=1, filter_goal_mets=False):
        indices = np.random.randint(0, self._size, batch_size)
        out = dict()
        for k, v in self._buffers.items():
            samples = v[indices, -horizon:, ...]
            samples = samples.squeeze(axis=1) if samples.shape[1] == 1 else samples
            out[k] = samples
        if filter_goal_mets:
            goal_mets = np.array(list(map(lambda info: info.get('goal_met', False), out['info'])))
            # We mask transitions with 'goal_met' since they are non-continuous, what extremely
            # destabilizes the learning of p(s_t_1 | s_t, a_t)
            out = {k: v[~goal_mets, ...] for k, v in out.items()}
        return {k: v for k, v in out.items() if k != 'info'}

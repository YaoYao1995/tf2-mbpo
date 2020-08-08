import numpy as np


class ReplayBuffer(object):
    def __init__(self, observation_dim, action_dim, buffer_capacity=1000000):
        self._buffer_capacity = buffer_capacity
        self._buffers = {
            'observation': np.empty((buffer_capacity, observation_dim),
                                    dtype=np.float32),
            'next_observation': np.empty((buffer_capacity, observation_dim),
                                         dtype=np.float32),
            'action': np.empty((buffer_capacity, action_dim), dtype=np.float32),
            'reward': np.empty((buffer_capacity, 1), dtype=np.float32),
            'terminal': np.empty((buffer_capacity, 1), dtype=np.bool),
            'info': np.empty((buffer_capacity, 1), dtype=dict)
        }
        self._size = 0
        self._ptr = 0
        self.obs_mean = np.zeros((observation_dim,))
        self.obs_stddev = np.ones_like(self.obs_mean)

    def update_statistics(self):
        cat = np.concatenate([
            self._buffers['observation'], self._buffers['next_observation']], axis=0)
        self.obs_mean = np.mean(cat, axis=0)
        self.obs_stddev = np.std(cat, axis=0)

    def store(self, transition):
        self._size = min(self._size + 1, self._buffer_capacity)
        for k, v in transition.items():
            self._buffers[k][self._ptr:self._ptr + 1, ...] = transition[k]
        self._ptr = (self._ptr + 1) % self._buffer_capacity

    def sample(self, batch_size, filter_goal_mets=False):
        indices = np.random.randint(0, self._size, batch_size)
        out = {k: v[indices, ...] for k, v in self._buffers.items()}
        if filter_goal_mets:
            goal_mets = np.array(list(map(lambda info: info.get('goal_met', False), out['info'])))
            # We mask transitions with 'goal_met' since they are non-continuous, what extremely
            # destabilizes the learning of p(s_t_1 | s_t, a_t)
            out = {k: v[~goal_mets, ...] for k, v in out.items()}
        out['observation'] = np.clip((out['observation'] - self.obs_mean) / self.obs_stddev,
                                     -10.0, 10.0)
        out['next_observation'] = np.clip(
            (out['next_observation'] - self.obs_mean) / self.obs_stddev,
            -10.0, 10.0)
        return {k: v for k, v in out.items() if k != 'info'}

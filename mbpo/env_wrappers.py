import numpy as np
from gym import Wrapper, ObservationWrapper
from gym.spaces import Box

from gym.wrappers import Monitor


# Copied from https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c
# /wrappers.py
class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, 'Expects at least one repeat.'
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class ObservationNormalize(ObservationWrapper):
    def __init__(self, env):
        super(ObservationNormalize, self).__init__(env)
        # Computing running mean and variance with
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford
        # 's_online_algorithm
        self._step = 0
        self._mean = np.zeros(self.env.observation_space.shape,
                              dtype=self.env.observation_space.dtype)
        self._m2 = np.ones(self.env.observation_space.shape,
                           dtype=self.env.observation_space.dtype)
        self._mask = np.logical_and(
            np.isfinite(env.observation_space.low),
            np.isfinite(env.observation_space.high)
        )
        # For a-priori known scales of the observation space (i.e., finite), we normalize between
        # -1 and 1.
        self.observation_space = Box(
            low=np.where(self._mask, -np.ones_like(self.env.observation_space.low),
                         self.env.observation_space.low),
            high=np.where(self._mask, np.ones_like(self.env.observation_space.high),
                          self.env.observation_space.high),
            dtype=env.observation_space.dtype)

    def normalize(self, observation):
        # s^2 = m2 / (n - 1), sigma^2 = m2 / n, where s^2 is an *unbiased* estimate of the variance.
        # We devide m2 by max(step - 1, 1), so that at step = 0, we don't divide by
        # zero but just get a biased estimate of the variance.
        return np.clip(np.where(
            self._mask,
            (self.observation_space.high - self.observation_space.low) /
            (self.env.observation_space.high - self.env.observation_space.low) *
            (observation - self.env.observation_space.low) + self.observation_space.low,
            (observation - self._mean) / (np.sqrt(self._m2 / (max(self._step - 1.0, 1.0))) + 1e-8)
        ), -10.0, 10.0)

    def observation(self, observation):
        self._step += 1
        delta = observation - self._mean
        self._mean += delta / self._step
        self._m2 += delta * (observation - self._mean)
        return self.normalize(observation)


# Normalized observations using the mean and variance of data collected in training
# without updating the statistics of test data.
class TestObservationNormalize(ObservationWrapper):
    def __init__(self, env):
        assert isinstance(env, ObservationNormalize), \
            "TestObservationNormalize can only wrap ObservationNormalize"
        super(TestObservationNormalize, self).__init__(env)

    def observation(self, observation):
        return self.env.normalize(observation)

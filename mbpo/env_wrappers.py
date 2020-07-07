from gym import Wrapper


# Copied from https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c
# /wrappers.py
class ActionRepeatWrapper(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, 'Expects at least one repeat.'
        super(ActionRepeatWrapper, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        current_step = 0.0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

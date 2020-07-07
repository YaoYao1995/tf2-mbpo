import numpy as np
import tensorflow as tf

import mbpo.models as models
from mbpo.replay_buffer import ReplayBuffer


class MBPO(tf.Module):
    def __init__(self, config, writer, environment_data, observation_space, action_space):
        super(MBPO, self).__init__()
        self._config = config
        self._writer = writer
        self._training_step = 0
        self._model_data = ReplayBuffer(observation_space.shape[0], action_space.shape[0])
        self._environment_data = environment_data
        self._ensemble = [models.WorldModel(
            observation_space.shape[0],
            self._config.dynamics_layers,
            self._config.units, reward_layers=2, terminal_layers=2) for _ in
            range(self._config.ensemble_size)]
        self._model_optimizer = tf.keras.optimizers.Adam(
            self._config.model_learning_rate, clipnorm=self._config.clip_norm, epsilon=1e-5
        )
        self._warmup_policy = lambda: action_space.sample()
        self._actor = models.Actor(action_space.shape[0], 3, self._config.units)
        self._critic_target = models.Critic(3, self._config.units)
        self._critic_king = models.Critic(3, self._config.units)
        self._critic_queen = models.Critic(3, self._config.units)

    def imagine_rollouts(self, sampled_observations):
        rollouts = {'observation': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'next_observation': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'action': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'reward': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'terminal': tf.TensorArray(tf.bool, size=self._config.horizon)}
        observation = sampled_observations
        for k in tf.range(self._config.horizon):
            rollouts['observation'] = rollouts['observation'].write(k, observation)
            action = self._actor(tf.stop_gradient(observation))
            rollouts['action'] = rollouts['action'].write(k, action)
            bootstrap = tf.random.uniform((1,), maxval=self._config.ensemble_size, dtype=tf.int32)
            predictions = self._ensemble[bootstrap](
                observation, action)
            observation = predictions['next_observation'].sample()
            rollouts['next_observation'] = rollouts['next_observation'].write(k, observation)
            rollouts['reward'] = rollouts['reward'].write(k, predictions['reward'].mean())
            # TODO (yarden): understand if it's better to keep the probs (from STEVE paper).
            rollouts['terminal'] = rollouts['terminal'].write(k, tf.greater_equal(
                predictions['terminal'].probs, 0.5))
        return {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in rollouts.items()}

    def update_model(self):
        for _ in range(self._config.model_grad_steps):
            observations, next_observations, actions, rewards, terminals, infos = \
                self._environment_data.sample(
                    self._config.model_batch_size * self._config.ensemble_size)
            goal_mets = np.array(list(map(lambda info: info.get('goal_met', False), infos)))
            # We mask transitions with 'goal_met' since they are non-continuous, what extremely
            # destabilizes the learning of p(s_t_1 | s_t, a_t)
            masked_observations, masked_actions, masked_next_observations, masked_rewards = \
                observations[~goal_mets, ...], actions[~goal_mets, ...], next_observations[
                    ~goal_mets, ...], rewards[~goal_mets, ...]
            batch = {'observations': observations,
                     'next_observations': next_observations,
                     'actions': actions,
                     'rewards': rewards,
                     'terminals': terminals}
            self._model_training_step(batch)

    def _model_training_step(self, batch):
        bootstraps_batches = {k: tf.split(v, [
            tf.shape(batch)[0] // self._config.ensemble_size] * self._config.ensemble_size) for k, v
                              in batch.items()}
        parameters = []
        loss = 0.0
        with tf.GradientTape() as model_tape:
            for i, bootstrap in enumerate(self._ensemble):
                observations, next_observations, actions, rewards, terminals = \
                bootstraps_batches['observations'][i], bootstraps_batches['next_observations'][i], \
                bootstraps_batches['actions'][i], bootstraps_batches['rewards'][i], \
                bootstraps_batches['terminals'][i]
                predictions = bootstrap(observations, actions)
                log_p_dynamics = tf.reduce_mean(
                    predictions['next_observation'].log_prob(next_observations))
                log_p_reward = tf.reduce_mean(predictions['reward'].log_prob(rewards))
                log_p_terminals = tf.reduce_mean(predictions['terminal'].log_prob(terminals))
                loss -= (log_p_dynamics + log_p_reward + log_p_terminals)
                parameters += bootstrap.trainable_variables
            grads = model_tape.gradient(loss, parameters)
            self._model_optimizer.apply_gradients(zip(grads, parameters))
        # TODO (yarden): log scalars here.

    def update_actor_critic(self):
        # TODO (yarde): grad clip and L2 penalty
        pass

    def _write_summary(self):
        pass

    @property
    def time_to_update_model(self):
        return self._training_step and self._training_step % self._config.steps_per_epoch == 0

    @property
    def time_to_log(self):
        return self._training_step and self._training_step % self._config.steps_per_log == 0

    @property
    def warm(self):
        return self._training_step >= self._config.warmup_training_steps

    def __call__(self, observation, training=True):
        if training:
            if self.time_to_update_model:
                self.update_model()
            if self.warm:
                action = self._actor(observation).sample()
                sampled_observations, *_ = self._model_data.sample(self._config.model_rollouts)
                self._model_data.store(self.imagine_rollouts(sampled_observations))
                self.update_actor_critic()
            else:
                action = self._warmup_policy()
            self._training_step += self._config.action_repeat
        else:
            action = self._actor(observation).mode()
        if self.time_to_log:
            self._write_summary()
        return action

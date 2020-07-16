import tensorflow as tf
from tensorflow_addons.optimizers import AdamW

import mbpo.models as models
from mbpo.replay_buffer import ReplayBuffer


class MBPO(tf.Module):
    def __init__(self, config, logger, observation_space, action_space):
        super(MBPO, self).__init__()
        self._config = config
        self._logger = logger
        self._training_step = 0
        self._model_data = ReplayBuffer(observation_space.shape[0], action_space.shape[0])
        self._environment_data = ReplayBuffer(observation_space.shape[0], action_space.shape[0])
        self._ensemble = [models.WorldModel(
            observation_space.shape[0],
            self._config.dynamics_layers,
            self._config.units, reward_layers=2, terminal_layers=2)
            for _ in range(self._config.ensemble_size)]
        self._model_optimizer = AdamW(
            self._config.model_learning_rate, clipnorm=self._config.clip_norm, epsilon=1e-5,
            weight_decay=self._config.weight_decay
        )
        self._warmup_policy = lambda: action_space.sample()
        self._actor = models.Actor(action_space.shape[0], 3, self._config.units)
        self._actor_optimizer = AdamW(
            self._config.actor_learning_rate, clipnorm=self._config.clip_norm, epsilon=1e-5,
            weight_decay=self._config.weight_decay
        )
        self._critic = models.Critic(3, self._config.units)
        self._critic_optimizer = AdamW(
            self._config.critic_learning_rate, clipnorm=self._config.clip_norm, epsilon=1e-5,
            weight_decay=self._config.weight_decay
        )

    def imagine_rollouts(self, sampled_observations):
        rollouts = {'observation': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'next_observation': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'action': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'reward': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'terminal': tf.TensorArray(tf.bool, size=self._config.horizon)}
        done_rollouts = tf.zeros((tf.shape(sampled_observations)[0], self._config.horizon),
                                 dtype=tf.bool)
        observation = sampled_observations
        for k in tf.range(self._config.horizon):
            rollouts['observation'] = rollouts['observation'].write(k, observation)
            action = self._actor(tf.stop_gradient(observation)).sample()
            rollouts['action'] = rollouts['action'].write(k, action)
            bootstrap = tf.random.uniform((1,), maxval=self._config.ensemble_size, dtype=tf.int32)
            predictions = self._ensemble[bootstrap](observation, action)
            observation = predictions['next_observation'].sample()
            rollouts['next_observation'] = rollouts['next_observation'].write(k, observation)
            rollouts['reward'] = rollouts['reward'].write(k, predictions['reward'].mean())
            # TODO (yarden): understand if it's better to keep the probs (from STEVE paper).
            terminal = tf.greater_equal(predictions['terminal'].probs, 0.5)
            if k < self._config.horizon - 1:
                done_rollouts[:, k + 1] = tf.logical_or(
                    done_rollouts[:, k], terminal)
            rollouts['terminal'] = rollouts['terminal'].write(k, terminal)
        done_rollouts_mask = tf.logical_not(done_rollouts)

        def filter_steps_after_terminal(unfiltered_rollouts, rollouts_masks):
            return tf.map_fn(
                lambda mask_and_rollout: tf.boolean_mask(mask_and_rollout[0], mask_and_rollout[1]),
                (rollouts_masks, tf.transpose(unfiltered_rollouts.stack(), [1, 0, 2])),
                dtype=unfiltered_rollouts.dtype)

        return {k: filter_steps_after_terminal(v, done_rollouts_mask) for k, v in rollouts}

    def update_model(self):
        for _ in range(self._config.model_grad_steps):
            batch = self._environment_data.sample(
                self._config.model_batch_size * self._config.ensemble_size,
                filter_goal_mets=True)
            self._model_training_step(batch)

    def _model_training_step(self, batch):
        bootstraps_batches = {k: tf.split(
            v, [tf.shape(batch)[0] // self._config.ensemble_size] * self._config.ensemble_size)
            for k, v in batch.items()}
        parameters = []
        loss = 0.0
        with tf.GradientTape() as model_tape:
            for i, world_model in enumerate(self._ensemble):
                observations, target_next_observations, \
                actions, target_rewards, target_terminals = \
                    bootstraps_batches['observations'][i], \
                    bootstraps_batches['next_observations'][i], \
                    bootstraps_batches['actions'][i], bootstraps_batches['rewards'][i], \
                    bootstraps_batches['terminals'][i]
                predictions = world_model(observations, actions)
                log_p_dynamics = tf.reduce_mean(
                    predictions['next_observation'].log_prob(target_next_observations))
                log_p_reward = tf.reduce_mean(predictions['reward'].log_prob(target_rewards))
                log_p_terminals = tf.reduce_mean(predictions['terminal'].log_prob(target_terminals))
                loss -= (log_p_dynamics + log_p_reward + log_p_terminals)
                parameters += world_model.trainable_variables
                self._logger['dynamics_' + str(i) + '_log_p'].update_state(-log_p_dynamics)
                self._logger['rewards_' + str(i) + '_log_p'].update_state(-log_p_reward)
                self._logger['terminals_' + str(i) + '_log_p'].update_state(-log_p_terminals)
            grads = model_tape.gradient(loss, parameters)
            self._model_optimizer.apply_gradients(zip(grads, parameters))
        self._logger['world_model_total_loss'].update_state(loss)
        self._logger['world_model_grads'].update_state(tf.norm(grads))

    def update_actor_critic(self):
        for _ in range(self._config.actor_critic_grad_steps):
            batch = self._model_data.sample(self._config.actor_critic_batch_size)
            actor_loss, actor_grads, pi_entropy = self._actor_training_step(batch)
            critic_loss, critic_grads = self._critic_training_step(batch)
            self._logger['actor_loss'].update_state(actor_loss)
            self._logger['actor_grads'].update_state(tf.norm(actor_grads))
            self._logger['critic_loss'].update_state(critic_loss)
            self._logger['critic_grads'].update_state(tf.norm(critic_grads))
            self._logger['pi_entropy'].update_state(pi_entropy)

    def _actor_training_step(self, batch):
        with tf.GradientTape() as actor_tape:
            pi = self._actor(batch['observation'])
            actor_loss = -tf.reduce_mean(
                self._critic(batch['observation'], pi.sample()).mode())
            grads = actor_tape.gradient(actor_loss, self._actor.trainable_variables)
            self._actor_optimizer.apply_gradients(zip(grads, self._actor.trainable_variables))
        return actor_loss, tf.norm(grads), pi.entropy()

    def _critic_training_step(self, batch):
        with tf.GradientTape as critic_tape:
            observations, actions, rewards, terminals = batch['observation'], batch['action'], \
                                                        batch['reward'], batch['terminal']
            td = rewards + self._config.discount * (1.0 - tf.cast(terminals, tf.float32)) * \
                 self._critic(observations, actions).mode()
            value_log_p = self._critic.log_prob(tf.stop_gradient(td))
            grads = critic_tape.gradient(-value_log_p, self._critic.trainable_variables)
            self._critic_optimizer.apply_gradients(zip(grads, self._critic.trainable_variables))
        return value_log_p, tf.norm(grads)

    @property
    def time_to_update_model(self):
        return self._training_step and self._training_step % self._config.steps_per_epoch == 0

    @property
    def time_to_log(self):
        return self._training_step and self._training_step % self._config.steps_per_log == 0

    @property
    def warm(self):
        return self._training_step >= self._config.warmup_training_steps

    def observe(self, transition):
        self._environment_data.store(transition)

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
            self._logger.log_metrics(self._training_step)
        return action

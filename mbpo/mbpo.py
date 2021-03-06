import random

import numpy as np
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
        self._model_data = ReplayBuffer(
            observation_space.shape[0], action_space.shape[0], self._config.horizon)
        self._environment_data = ReplayBuffer(
            observation_space.shape[0], action_space.shape[0], 1)
        self._ensemble = [models.WorldModel(
            observation_space.shape[0],
            self._config.dynamics_layers,
            self._config.units, reward_layers=2, terminal_layers=2)
            for _ in range(self._config.ensemble_size)]
        self._model_optimizer = AdamW(
            learning_rate=self._config.model_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5, weight_decay=self._config.weight_decay
        )
        self._warmup_policy = lambda: action_space.sample()
        self._actor = models.Actor(action_space.shape[0], 3, self._config.units)
        self._actor_optimizer = AdamW(
            learning_rate=self._config.actor_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5, weight_decay=self._config.weight_decay
        )
        self._critic = models.Critic(
            3, self._config.units, output_regularization=self._config.critic_regularization)
        self._critic_optimizer = AdamW(
            learning_rate=self._config.critic_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5, weight_decay=self._config.weight_decay
        )

    @tf.function
    def imagine_rollouts(self, sampled_observations, bootstrap, actions=None):
        rollouts = {'observation': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'next_observation': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'action': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'reward': tf.TensorArray(tf.float32, size=self._config.horizon),
                    'terminal': tf.TensorArray(tf.bool, size=self._config.horizon)}
        done_rollout = tf.zeros((tf.shape(sampled_observations)[0],), dtype=tf.bool)
        observation = sampled_observations
        for k in tf.range(self._config.horizon):
            rollouts['observation'] = rollouts['observation'].write(k, observation)
            action = self._actor(tf.stop_gradient(observation)).sample() \
                if actions is None else actions[k]
            rollouts['action'] = rollouts['action'].write(k, action)
            predictions = bootstrap(observation, action)
            # If the rollout is done, we stay at the terminal state, not overriding with a new,
            # possibly valid state.
            observation = tf.where(done_rollout,
                                   observation,
                                   predictions['next_observation'].sample())
            rollouts['next_observation'] = rollouts['next_observation'].write(k, observation)
            terminal = tf.where(done_rollout,
                                True,
                                tf.cast(predictions['terminal'].mode(), tf.bool))
            rollouts['terminal'] = rollouts['terminal'].write(k, terminal)
            reward = tf.where(done_rollout,
                              0.0,
                              predictions['reward'].mean())
            rollouts['reward'] = rollouts['reward'].write(k, reward)
            done_rollout = tf.logical_or(
                terminal, done_rollout)
        return {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in rollouts.items()}

    def update_model(self):
        print("Updating model.")
        for _ in range(self._config.model_grad_steps):
            batch = self._environment_data.sample(
                self._config.model_batch_size * self._config.ensemble_size,
                filter_goal_mets=self._config.filter_goal_mets)
            self._model_grad_step(batch)
        print("Model update done.")

    @tf.function
    def _model_grad_step(self, batch):
        bootstraps_batches = {k: tf.split(
            v, [tf.shape(batch['observation'])[0] // self._config.ensemble_size] *
               self._config.ensemble_size) for k, v in batch.items()}
        parameters = []
        loss = 0.0
        with tf.GradientTape() as model_tape:
            for i, world_model in enumerate(self._ensemble):
                observations, target_next_observations, \
                actions, target_rewards, target_terminals = \
                    bootstraps_batches['observation'][i], \
                    bootstraps_batches['next_observation'][i], \
                    bootstraps_batches['action'][i], bootstraps_batches['reward'][i], \
                    bootstraps_batches['terminal'][i]
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
        self._logger['world_model_grads'].update_state(tf.linalg.global_norm(grads))

    def update_actor_critic(self):
        for _ in range(self._config.actor_critic_update_steps):
            batch = self._model_data.sample(self._config.actor_critic_batch_size)
            self._update_critic(
                tf.constant(batch['observation'], dtype=tf.float32),
                tf.constant(batch['next_observation'], dtype=tf.float32),
                tf.constant(batch['action'], dtype=tf.float32),
                tf.constant(batch['reward'], dtype=tf.float32),
                tf.constant(batch['terminal'], dtype=tf.float32)
            )
            actor_loss, actor_grads, pi_entropy = self._actor_grad_step(
                tf.constant(batch['observation']))
            self._logger['actor_loss'].update_state(actor_loss)
            self._logger['actor_grads'].update_state(tf.norm(actor_grads))
            self._logger['pi_entropy'].update_state(pi_entropy)

    @tf.function
    def _update_critic(self, observation, next_observation, action, reward, terminal):
        bootsrapped_target = reward + self._config.discount * (1.0 - terminal) * \
                    self._critic(next_observation,
                                 self._actor(next_observation).sample()).mode()
        critic_loss, critic_grads = self._critic_grad_step(
            observation,
            action,
            bootsrapped_target)
        self._logger['critic_loss'].update_state(critic_loss)
        self._logger['critic_grads'].update_state(tf.norm(critic_grads))

    @tf.function
    def _critic_grad_step(self, observation, action, bootsrapped_target):
        with tf.GradientTape() as critic_tape:
            q_log_p = self._critic(observation, action)\
                .log_prob(tf.stop_gradient(bootsrapped_target))
            grads = critic_tape.gradient(-q_log_p, self._critic.trainable_variables)
            self._critic_optimizer.apply_gradients(zip(grads, self._critic.trainable_variables))
        return -q_log_p, tf.linalg.global_norm(grads)

    @tf.function
    def _actor_grad_step(self, observation):
        with tf.GradientTape() as actor_tape:
            pi = self._actor(observation)
            actor_loss = -tf.reduce_mean(
                self._critic(observation, pi.sample()).mode())
            grads = actor_tape.gradient(actor_loss, self._actor.trainable_variables)
            self._actor_optimizer.apply_gradients(zip(grads, self._actor.trainable_variables))
        return actor_loss, tf.linalg.global_norm(grads), pi.entropy()

    @property
    def time_to_update_model(self):
        return self._training_step and \
               self._training_step % self._config.steps_per_model_update < \
               self._config.action_repeat

    @property
    def time_to_log(self):
        return self._training_step and \
               self._training_step % self._config.steps_per_log < self._config.action_repeat

    @property
    def warm(self):
        return self._training_step >= self._config.warmup_training_steps

    def observe(self, transition):
        self._environment_data.store(transition)

    def __call__(self, observation, training=True):
        if training:
            if self.time_to_update_model and self.warm:
                self.update_model()
            if self.warm:
                action = self._actor(
                    np.expand_dims(observation, axis=0).astype(np.float32)).sample().numpy()
                real_transitions = self._environment_data.sample(self._config.model_rollouts)
                self._model_data.store(self.imagine_rollouts(
                    tf.constant(real_transitions['observation']),
                    random.choice(self._ensemble)))
                self.update_actor_critic()
            else:
                action = self._warmup_policy()
            self._training_step += self._config.action_repeat
        else:
            action = self._actor(
                np.expand_dims(observation, axis=0).astype(np.float32)).mode().numpy()
        if self.time_to_log and training and self.warm:
            self._logger.log_metrics(self._training_step)
        return action

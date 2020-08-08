import random

import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
from tqdm import tqdm

import mbpo.models as models
from mbpo.replay_buffer import ReplayBuffer


# TODO (yarden): observe gradients of actor - if small maybe vanishing??? - if so, return to
# DDPG/SAC style algorithms (with q-function) or use gru/lstm in dynamics

class MBPO(tf.Module):
    def __init__(self, config, logger, observation_space, action_space):
        super(MBPO, self).__init__()
        self._config = config
        self._logger = logger
        self._training_step = 0
        self._experience = ReplayBuffer(observation_space.shape[0], action_space.shape[0])
        self.ensemble = [models.WorldModel(
            observation_space.shape[0],
            self._config.dynamics_layers,
            self._config.units, reward_layers=2, terminal_layers=2)
            for _ in range(self._config.ensemble_size)]
        self._model_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.model_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5
        )
        self._warmup_policy = lambda: action_space.sample()
        self._actor = models.Actor(action_space.shape[0], 3, self._config.units)
        self._actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.actor_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5
        )
        self._critic = models.Critic(
            3, self._config.units, output_regularization=self._config.critic_regularization)
        self._critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.critic_learning_rate, clipnorm=self._config.grad_clip_norm,
            epsilon=1e-5
        )

    def update_model(self, batch):
        self._model_grad_step(batch)

    @tf.function
    def _model_grad_step(self, batch):
        bootstraps_batches = {k: tf.split(
            v, [tf.shape(batch['observation'])[0] // self._config.ensemble_size] *
               self._config.ensemble_size +
               [tf.shape(batch['observation'])[0] % self._config.ensemble_size])
            for k, v in batch.items()}
        parameters = []
        loss = 0.0
        with tf.GradientTape() as model_tape:
            for i, world_model in enumerate(self.ensemble):
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

    def imagine_rollouts(self, sampled_observations, bootstrap, actions=None):
        horizon = self._config.horizon if actions is None else tf.shape(actions)[0]
        rollouts = {'observation': tf.TensorArray(tf.float32, size=horizon),
                    'next_observation': tf.TensorArray(tf.float32, size=horizon),
                    'action': tf.TensorArray(tf.float32, size=horizon),
                    'reward': tf.TensorArray(tf.float32, size=horizon),
                    'terminal': tf.TensorArray(tf.float32, size=horizon)}
        done_rollout = tf.zeros((tf.shape(sampled_observations)[0], 1), dtype=tf.bool)
        observation = sampled_observations
        for k in tf.range(horizon if actions is None else tf.shape(actions)[0]):
            rollouts['observation'] = rollouts['observation'].write(k, observation)
            action = self._actor(tf.stop_gradient(observation)).sample() \
                if actions is None else actions[k, ...]
            rollouts['action'] = rollouts['action'].write(k, action)
            predictions = bootstrap(observation, action)
            # If the rollout is done, we stay at the terminal state, not overriding with a new,
            # possibly valid state.
            observation = tf.where(done_rollout,
                                   observation,
                                   predictions['next_observation'].sample())
            rollouts['next_observation'] = rollouts['next_observation'].write(k, observation)
            terminal = tf.where(done_rollout,
                                1.0,
                                predictions['terminal'].mode())
            rollouts['terminal'] = rollouts['terminal'].write(k, terminal)
            reward = tf.where(done_rollout,
                              0.0,
                              predictions['reward'].mode())
            rollouts['reward'] = rollouts['reward'].write(k, reward)
            done_rollout = tf.logical_or(
                tf.cast(terminal, tf.bool), done_rollout)
        return {k: tf.transpose(v.stack(), [1, 0, 2]) for k, v in rollouts.items()}

    def compute_lambda_values(self, rollouts):
        next_observation, actions, rewards, terminals = rollouts['next_observation'], \
                                                        rollouts['action'], \
                                                        rollouts['reward'], \
                                                        rollouts['terminal']
        lambda_values = tf.TensorArray(tf.float32, self._config.horizon)
        v_lambda = self._critic(next_observation[:, -1, ...]).mode() * (1.0 - terminals[:, -1])
        # reverse traversing over data.
        for t in tf.range(self._config.horizon - 1, -1, -1):
            td = rewards[:, t] + \
                 (1.0 - terminals[:, t]) * (1.0 - self._config.lambda_) * \
                 self._config.discount * self._critic(next_observation[:, t, ...]).mode()
            v_lambda = td + v_lambda * self._config.lambda_ * self._config.discount
            lambda_values = lambda_values.write(t, v_lambda)
        return tf.transpose(lambda_values.stack(), [1, 0, 2])

    @tf.function
    def update_actor_critic(self, observation, model_bootstrap):
        with tf.GradientTape() as actor_tape:
            imagined_rollouts = self.imagine_rollouts(observation, model_bootstrap)
            lambda_values = self.compute_lambda_values(imagined_rollouts)
            actor_loss, actor_grads = self._actor_grad_step(
                lambda_values, imagined_rollouts['terminal'], actor_tape)
        with tf.GradientTape() as critic_tape:
            critic_loss, critic_grads = self._critic_grad_step(
                lambda_values, imagined_rollouts['observation'], imagined_rollouts['terminal'],
                critic_tape)
        self._logger['actor_loss'].update_state(actor_loss)
        self._logger['actor_grads'].update_state(tf.norm(actor_grads))
        self._logger['critic_loss'].update_state(critic_loss)
        self._logger['critic_grads'].update_state(tf.norm(critic_grads))
        self._logger['pi_entropy'].update_state(self._actor(observation).entropy())

    def _actor_grad_step(self, lambda_values, terminals, actor_tape):
        actor_loss = -tf.reduce_mean(
            tf.reduce_sum(lambda_values * (1.0 - terminals), axis=1))
        grads = actor_tape.gradient(actor_loss, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(zip(grads, self._actor.trainable_variables))
        return actor_loss, tf.linalg.global_norm(grads)

    def _critic_grad_step(self, lambda_values, observations, terminals, critic_tape):
        critic_loss = 0.0
        for t in tf.range(self._config.horizon):
            critic_loss -= tf.reduce_mean(self._critic(observations[:, t, ...])
                                          .log_prob(tf.stop_gradient(lambda_values[:, t, ...])) *
                                          tf.stop_gradient(1.0 - terminals[:, t, ...]))
        grads = critic_tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(grads, self._critic.trainable_variables))
        return critic_loss, tf.linalg.global_norm(grads)

    @property
    def time_to_update(self):
        return self._training_step and \
               self._training_step % self._config.steps_per_update < self._config.action_repeat

    @property
    def time_to_log(self):
        return self._training_step and \
               self._training_step % self._config.steps_per_log < self._config.action_repeat

    @property
    def warm(self):
        return self._training_step >= self._config.warmup_training_steps

    def observe(self, transition):
        self._training_step += transition.pop('steps', self._config.action_repeat)
        self._experience.store(transition)

    def __call__(self, observation, training=True):
        if training:
            if self.warm:
                # action = self._actor(
                #     np.expand_dims(observation, axis=0).astype(np.float32)).sample().numpy()
                action = self.debug_actor(tf.constant(observation, dtype=tf.float32)).numpy()
            else:
                action = self._warmup_policy()
            if self.time_to_update and self.warm:
                print("Updating world model, actor and critic.")
                for _ in tqdm(range(self._config.update_steps), position=0, leave=True):
                    batch = self._experience.sample(self._config.batch_size,
                                                    filter_goal_mets=self._config.filter_goal_mets)
                    self.update_model(batch)
                    self.update_actor_critic(
                        tf.constant(batch['observation'], dtype=tf.float32),
                        random.choice(self.ensemble))
        else:
            # action = self._actor(
            #     np.expand_dims(observation, axis=0).astype(np.float32)).mode().numpy()
            action = self.debug_actor(tf.constant(observation, dtype=tf.float32)).numpy()
        if self.time_to_log and training and self.warm:
            self._logger.log_metrics(self._training_step)
        return action

    @tf.function
    def debug_actor(self, observation):
        action_dim = 1
        mu = tf.zeros((8, action_dim))
        sigma = tf.ones_like(mu)
        best_so_far = tf.zeros((action_dim,), dtype=tf.float32)
        best_so_far_score = -np.inf * tf.ones((), dtype=tf.float32)
        for _ in tf.range(10):
            action_sequences = tf.random.normal(
                shape=(150, 8, action_dim),
                mean=mu, stddev=sigma
            )
        # TODO (yarden): average over particles!!!
            action_sequences = tf.clip_by_value(action_sequences, -1.0, 1.0)
            action_sequences_batch = action_sequences
            all_rewards = []
            for model in self.ensemble:
                trajectories = self.imagine_rollouts(
                    tf.broadcast_to(observation, (action_sequences_batch.shape[0], observation.shape[0])),
                    model,
                    tf.transpose(action_sequences_batch, [1, 0, 2])
                )
                all_rewards.append(tf.reduce_sum(
                    trajectories['reward'] * (1.0 - trajectories['terminal']), axis=1))
            scores = tf.squeeze(tf.reduce_mean(tf.stack(all_rewards, axis=0), axis=0))
            elite_scores, elite = tf.nn.top_k(scores, 10, sorted=False)
            best_of_elite = tf.argmax(elite_scores)
            if tf.greater(elite_scores[best_of_elite], best_so_far_score):
                best_so_far = action_sequences[elite[best_of_elite], 0, ...]
                best_so_far_score = elite_scores[best_of_elite]
            elite_actions = tf.gather(action_sequences, elite, axis=0)
            mean, variance = tf.nn.moments(elite_actions, axes=0)
            mu = mean
            sigma = tf.sqrt(variance)
            if tf.less_equal(tf.reduce_mean(sigma), 0.1):
                break
        return tf.clip_by_value(best_so_far + tf.random.normal(best_so_far.shape, stddev=0.01), -1.0, 1.0)

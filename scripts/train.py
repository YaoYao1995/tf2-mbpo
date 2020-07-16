import argparse
from collections import defaultdict

from tqdm import tqdm


def do_episode(agent, training, environment, config, pbar, render):
    observation = environment.reset()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation, training)
        next_observation, reward, terminal, info = environment.step(action)
        agent.observe(dict(observation=observation,
                           next_observation=next_observation,
                           action=action,
                           reward=reward,
                           terminal=terminal,
                           info=info))
        if render:
            episode_summary['image'].append(environment.render())
        pbar.update(config.action_repeat)
        steps += config.action_repeat
        done = terminal or steps >= config.episode_length
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
        observation = next_observation
    return steps, episode_summary


def interact(agent, environment, steps, config, training=True):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []
    while steps_count < steps:
        episode_steps, episode_summary = \
            do_episode(agent, training,
                       environment, config,
                       pbar, len(episodes) < config.render_episodes)
        steps_count += episode_steps
        episodes.append(episode_summary)
    return steps, episodes


def define_config():
    pass


def main(config):
    steps = 0
    agent = None
    environment = None
    while steps < config.total_training_steps:
        training_steps, training_episodes_summaries = interact(
            agent, environment, config.training_steps, config, training=True)
        steps += training_steps
        evaluation_steps, evaluation_episodes_summaries = interact(
            agent, environment, config.eval_steps, config, training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument('--{}'.format(key), default=value)
    main(parser.parse_args())

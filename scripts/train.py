import argparse
from collections import defaultdict

from tqdm import tqdm


def do_episode(agent, training, environment, episode_length, pbar, render):
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
        pbar.update(1)
        steps += 1
        done = terminal or steps >= episode_length
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
        observation = next_observation
    return steps, episode_summary


def interact(agent, environment, steps, episode_length, training=True, render_episodes=0):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []
    while steps_count < steps:
        episode_steps, episode_summary = \
            do_episode(agent, training,
                       environment, episode_length,
                       pbar, len(episodes) < render_episodes)
        steps_count += episode_steps
        episodes.append(episode_summary)
    return episodes


def define_config():
    pass


def main(config):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument('--{}'.format(key), default=value)
    main(parser.parse_args())

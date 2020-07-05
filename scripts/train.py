import argparse
from mbpo.mbpo import MBPO


def define_config():
    pass


def main(config):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument('--{}'.format(key), default=value)
    main(parser.parse_args())
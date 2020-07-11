#!/usr/bin/env python

import sys

from setuptools import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6

setup(
    name='mbpo',
    packages=['mbpo'],
    install_requires=[
        'gym~=0.15.3',
        'mujoco_py==2.0.2.7',
        'numpy~=1.17.4',
        'tensorflow>=2.1.0',
        'tensorflow-probability',
        'moviepy>=1.0.0',
        'tensorboardx>=1.8'
    ]
)

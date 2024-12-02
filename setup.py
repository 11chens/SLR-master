from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.0',
    author='Nikita Rudin',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='rudinn@ethz.ch',
    description='Isaac Gym environments for Legged Robots',
    python_requires='>=3.6',
    install_requires=["isaacgym",
                      "matplotlib",
                      "torch>=1.4.0",
                      "torchvision>=0.5.0",
                      "numpy>=1.16.4"
            ]
)
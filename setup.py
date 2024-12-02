from setuptools import setup, find_packages

setup(
    name='SLR-master',  
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'isaacgym',
        'numpy==1.21',
        'tensorboard',
        'setuptools==59.5.0',
        'matplotlib',
        'opencv-contrib-python',
    ],
    python_requires='>=3.6', 
)

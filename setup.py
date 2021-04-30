import setuptools
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()
    
with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='blip-gw',
    version='1.0.6',
    author='Sharan Banagiri',
    author_email='banag002@umn.edu',
    description='A bayesian pipeline for detecting stochastic backgrounds with LISA.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'':['params.ini']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'blip = runblip.blip:main'
        ]
    },
    install_requires=requirements
)

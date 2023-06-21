#!/usr/bin/env python

import os

from setuptools import setup

with open("README.md", "r") as rme:
    long_description = rme.read()

setup(
    name="BLiP",
    description="A bayesian pipeline for detecting stochastic backgrounds with LISA.",
    long_description=long_description,
    url="https://github.com/sharanbngr/blip",
    author="Sharan Banagiri and others",
    author_email="sharan.banagiri@gmail.com",
    license="MIT",
    packages=["blip",
        "blip.src",
        "blip.tools",
        ],
    package_dir={"blip":"blip"},
    scripts=["blip/run_blip"],
    install_requires=[
        "numpy",
        "matplotlib",
        "healpy==1.15.2",
        "chainconsumer",
        "sympy",
        "legwork",
        "dill",
        "dynesty",
        "emcee",
        "nessai"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)



#!/usr/bin/env python
"""SimPEG: Simulation and Parameter Estimation in Geophysics

SimPEG is a python package for simulation and gradient based
parameter estimation in the context of geophysical applications.
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

setup(
    name="SimPEG",
    version="0.18.1",
    packages=find_packages(exclude=["tests*", "examples*", "tutorials*"]),
    install_requires=[
        "numpy>=1.7",
        "scipy>=1.0.0",
        "scikit-learn>=0.22",
        "pymatsolver>=0.1.1",
        "matplotlib",
        "discretize>=0.7.1",
        "geoana>=0.4.0",
        "empymod",
        "pandas",
    ],
    author="Rowan Cockett",
    author_email="rowanc1@gmail.com",
    description="SimPEG: Simulation and Parameter Estimation in Geophysics",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="geophysics inverse problem",
    url="http://simpeg.xyz/",
    download_url="http://github.com/simpeg/simpeg",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
)

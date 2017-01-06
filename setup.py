#!/usr/bin/env python

import os
from setuptools import setup, find_packages
from pip.req import parse_requirements


def get_requirements(fname):
    return [str(r.req) for r in parse_requirements(fname, session=False)]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="anneal",
    version="0.0",
    author="ohtaman",
    author_email="ohtamans@gmail.com",
    description="Simulated/Quantum Annealing Library",
    long_description=read('README.md'),
    url="http://www.github.com/ohtaman/anneal",
    license="MIT",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    setup_requires=get_requirements('requirements/setup.txt'),
    tests_require=get_requirements('requirements/test.txt')
)
import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))


def read(fname):
    return open(os.path.join(here, fname)).read()


readme = read('README.md')

setup(
    name="gatelfpytorchjson",
    version="0.2",
    description="Library to build and use pytorch NN models for GATE Learning Framework",
    author="Johann Petrak",
    author_email="johann.petrak@gmail.com",
    url="https://github.com/GateNLP/gate-lf-pytorch-json",
    license="Apache 2.0",
    # keywords="",
    packages=['gatelfpytorchjson'],
    long_description=readme,
    install_requires=['torch>=0.4.1', 'numpy'],
    python_requires=">=3.5",
    tests_require=['nose'],
    test_suite='nose.collector',
    classifiers=[],
)

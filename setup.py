"""
Just a regular `setup.py` file.

Author: Nikolay Lysenko
"""


import os
from setuptools import setup, find_packages


current_dir = os.path.abspath(os.path.dirname(__file__))

description = (
   'A virtual analog synthesizer that provides a flexible way to create '
   'new digital instruments with their own timbres.'
)
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sinethesizer',
    version='0.1.0',
    description=description,
    long_description=long_description,
    url='https://github.com/Nikolay-Lysenko/sinethesizer',
    author='Nikolay Lysenko',
    author_email='nikolay-lysenco@yandex.ru',
    license='MIT',
    keywords='synthesizer analog_synthesizer music timbre',
    packages=find_packages(exclude=['tests', 'docs']),
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy']
)

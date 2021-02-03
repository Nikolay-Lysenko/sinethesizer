"""
Just a regular `setup.py` file.

Author: Nikolay Lysenko
"""


import os
from setuptools import setup, find_packages


current_dir = os.path.abspath(os.path.dirname(__file__))

description = (
   'A standalone synthesizer that is controlled through text files '
   'in an extendable way.'
)
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sinethesizer',
    version='0.5.1',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Nikolay-Lysenko/sinethesizer',
    author='Nikolay Lysenko',
    author_email='nikolay-lysenco@yandex.ru',
    license='MIT',
    keywords=(
        'synthesizer additive_synthesis subtractive_synthesis fm_synthesis '
        'sound_effects adsr_envelope modulation_index filter_envelope ahdsr'
    ),
    packages=find_packages(),
    package_data={'sinethesizer': ['default_config.yml']},
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['numpy', 'pretty-midi', 'PyYAML', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)

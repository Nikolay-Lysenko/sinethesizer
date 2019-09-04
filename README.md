[![Build Status](https://travis-ci.org/Nikolay-Lysenko/sinethesizer.svg?branch=master)](https://travis-ci.org/Nikolay-Lysenko/sinethesizer)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/sinethesizer/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/sinethesizer)
[![Maintainability](https://api.codeclimate.com/v1/badges/a43618b5f9454d01186c/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/sinethesizer/maintainability)
[![PyPI version](https://badge.fury.io/py/sinethesizer.svg)](https://badge.fury.io/py/sinethesizer)

# [Sine]thesizer

## Overview

It is a digital additive synthesizer that provides a flexible way to create new virtual instruments with their own timbres.

The list of implemented and planned features is as follows:
- [x] Balance between freedom for user and simplicity of input formats
- [x] Stereo sound
- [x] Sound effects (e.g., vibrato, overdrive, phaser, etc)
- [x] Partial support of subtractive synthesis and FM synthesis
- [ ] Noises and drums
- [ ] Rich collection of presets

## Installation

To install a stable version, run:
```
pip install sinethesizer
```

## Usage

This synthesizer converts text files with parameters of sound events to WAV files with resulting audio tracks. It can be done with the following command:
```
python -m sinethesizer -i path/to/track.tsv -p path/to/presets.yml -o path/to/output.wav
```

Below table provides links to detailed information about input files that are required from user.

Option | Description | Example
:----: | :---------: | :-----:
-i path/to/track.tsv | [Track definition](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/track_definition.md) | [Scale](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/scale.tsv)
-p path/to/presets.yml | [Timbres definition](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/timbres_creation.md) | [Demo timbres](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/presets/demo.yml)

If something is still unclear, you can read the source code, because it is well-organized and has built-in documentation. Also your questions are welcome.

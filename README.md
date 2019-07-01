[![Maintainability](https://api.codeclimate.com/v1/badges/a43618b5f9454d01186c/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/sinethesizer/maintainability)
[![PyPI version](https://badge.fury.io/py/sinethesizer.svg)](https://badge.fury.io/py/sinethesizer)

# [Sine]thesizer

## Overview

It is a digital additive synthesizer that provides a flexible way to create new virtual instruments with their own timbres.

The list of implemented and planned features is as follows:
- [x] Balance between freedom for user and simplicity of input formats
- [x] Stereo sound
- [x] Sound effects (e.g., tremolo, vibrato, overdrive, etc)
- [ ] Noises and drums
- [ ] Rich collection of presets

## Installation

To install a stable version, run:
```
pip install sinethesizer
```

## Usage

To create a track, two things must be done:
* The track should be defined as [tab-separated file](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/scale.tsv);
* All used for this track virtual instruments should be [defined](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/presets/demo.yml).

Above links direct to simple examples that demonstrate how to do this. Anyway, for the sake of clarity, let us discuss some steps in details.

A tab-separated file with track definition has rows that represent sound events (loosely speaking, an event is a played note) and columns that represent properties of events. Required columns are as follows:

Column | Description
:-----: | :---------:
timbre | Name of a registered timbre
start_time | Time when event starts (in seconds)
duration | Duration of event (in seconds) including release stage
frequency | Frequency of sound (in Hz) or note (like A4); some timbres may ignore it
volume | Relative volume of the most loud piece of event
location | Position of sound source; a float between -1 and 1 where -1 stands for left channel only and 1 stands for right channel only
effects | List of effects in JSON; each record must have field "name" with registered effect name and, optionally, parameters of the effect; left this field blank if no effects are needed

After all preparations are done, synthesizer can be launched:
```
python -m sinethesizer -i path/to/file.tsv -o path/to/output.wav
```

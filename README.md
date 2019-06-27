[![PyPI version](https://badge.fury.io/py/sinethesizer.svg)](https://badge.fury.io/py/sinethesizer)

# [Sine]thesizer

## Overview

It is a virtual analog synthesizer that provides a flexible way to create new digital instruments with their own timbres.

The list of implemented and planned features is as follows:
- [x] High-level input format with enough freedom for user
- [x] Stereo sound
- [ ] Rich collection of presets
- [ ] Noises and drums
- [ ] Sound effects (e.g., filters, chorus, overdrive, etc)

## Installation

To install a stable version, run:
```
pip install sinethesizer
```

## Usage

To create a track, two things must be done:
* The track should be defined as [tab-separated file](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/scale.tsv);
* All used for this track timbres should be [defined](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/presets/basic_timbres.py) and [registered](https://github.com/Nikolay-Lysenko/sinethesizer//blob/master/sinethesizer/presets/registry.py).

Above links direct to simple examples that demonstrate how to do this. Anyway, for the sake of clarity, let us discuss some steps in details.

A tab-separated file with track definition has rows that represent sound events (loosely speaking, an event is a played note) and columns that represent properties of events. Required columns are as follows:

Column | Description
:-----: | :---------:
timbre | Name of a registered timbre
start_time | Time when event starts (in seconds)
duration | Duration of event (in seconds)
frequency | Frequency of sound (in Hz) or note (like A4); some timbres may ignore it
volume | Relative volume of the most loud piece of event
location | Position of sound source; a float between -1 and 1 where -1 stands for left channel only and 1 stands for right channel only

After all preparations are done, synthesizer can be launched:
```
python -m sinethesizer -i path/to/file.tsv -o path/to/output.wav
```

[![PyPI version](https://badge.fury.io/py/sinethesizer.svg)](https://badge.fury.io/py/sinethesizer)

# [Sine]thesizer

## Overview

It is a virtual analog synthesizer that provides a flexible way to create new digital instruments with their own timbres.

## Minimal Example

To create a track, two things must be done:
* The track should be defined as [JSON file](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/scale.json);
* All used for this track timbres should be [defined](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/presets/basic_timbres.py) and [registered](https://github.com/Nikolay-Lysenko/sinethesizer//blob/master/sinethesizer/presets/registry.py).

Above links direct to simple examples that demonstrate how to do this.

After all preparations are done, synthesizer can be launched:
```
python -m sinethesizer -i path/to/file.json -o path/to/output.wav
```

## Installation

To install a stable version, run:
```
pip install sinethesizer
```

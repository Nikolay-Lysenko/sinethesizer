[![Build Status](https://github.com/Nikolay-Lysenko/sinethesizer/actions/workflows/main.yml/badge.svg)](https://github.com/Nikolay-Lysenko/sinethesizer/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/sinethesizer/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/sinethesizer)
[![Maintainability](https://api.codeclimate.com/v1/badges/a43618b5f9454d01186c/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/sinethesizer/maintainability)
[![PyPI version](https://badge.fury.io/py/sinethesizer.svg)](https://badge.fury.io/py/sinethesizer)

# [Sine]thesizer

## Overview

It is a digital synthesizer that is based on some design principles:
* Control via text files facilitates automation and boosts reproducibility, so GUI is absent.
* Although low-level and OS-specific dependencies improve performance, they reduce reliability, portability, and transparency, so they are avoided here. This standalone synth depends only on Python and some its packages.
* Since performance is not a merit of this synth, it is better to trade off speedups for sound quality. In particular, wavetables are not used at all and full waves are generated. Also, noise is generated from scratch every time it is needed.

The list of implemented and planned features is as follows:
- [x] Balance between freedom for user and simplicity of input formats
- [x] Support of additive synthesis, subtractive synthesis, and AM/PM synthesis
- [x] Sound effects (e.g., phaser, overdrive, reverb, etc)
- [x] Custom envelopes
- [x] Noises and drums
- [ ] Rich collection of presets

## Installation

To install a stable version, run:
```
pip install sinethesizer
```

## Usage

This synthesizer converts MIDI files and special text files to WAV files with resulting audio tracks.

For a MIDI file, it can be done with the following command:
```bash
python -m sinethesizer \
    -i path/to/track.midi \
    -p path/to/presets.yml \  # Or -p path/to/dir_with_presets
    -m path/to/midi_config.yml \
    -o path/to/output.wav
```

However, MIDI files are binary and, therefore, quite opaque. Also, integration between them and this synth is not complete: for example, control changes are ignored and event-level effects can not be applied. Here, TSV (Tab-Separated Values) files of special schema can be used as a native and more transparent alternative to MIDI. To process such a file, run:
```bash
python -m sinethesizer \
    -i path/to/track.tsv \
    -p path/to/presets.yml \  # Or -p path/to/dir_with_presets
    -o path/to/output.wav
```

Below table provides links to detailed information about input files that are required from a user.

|           Option           |                                                    Description                                                     |                                                    Example                                                    |
|:--------------------------:|:------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|
|    -i path/to/track.tsv    |      [Track definition](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/track_definition.md)      |         [Scale](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/scale.tsv)          |
|   -p path/to/presets.yml   | [Instruments definition](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/instruments_creation.md) | [Demo instruments](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/instruments.yml) |
| -m path/to/midi_config.yml |                                        Settings of MIDI file interpretation                                        | [Demo MIDI config](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/midi_config.yml) |

If something is still unclear, you can read the source code — it is structured and has built-in documentation. Also, your questions are welcome.

## See also

To turn Jupyter notebook into a simple DAW, [PyMixer](https://github.com/Nikolay-Lysenko/pymixer) can be used. It is a Python library having good integration with [Sine]thesizer. Together they form a small ecosystem of audio tools which are oriented to text-based control.

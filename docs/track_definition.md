# Guide on Track Definition

A track to be recorded as WAV file should be defined as a MIDI file or a tab-separated file. If the latter option is chosen, a file must have rows that represent sound events (loosely speaking, an event is a played note) and columns that represent properties of events.

Required columns are as follows:

Column | Description
:-----: | :---------:
instrument | Name of instrument
start_time | Time when event starts (in seconds)
duration | Duration of event (in seconds) not including release stage and any prolongations caused by sound effects (such as reverb)
frequency | Frequency of sound (in Hz) or note (like A4, A#4, or Ab4); some instruments may ignore it
velocity | Force of sound generation; it can be likened to force of piano key pressing; it is a float between 0 and 1; it can affect volume and frequency spectrum
effects | List of [effects](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/effects/registry.py) in JSON; each record must have field "name" with supported effect name and, optionally, parameters of the effect; left this field blank if no effects are needed

For more intuitive explanation, look at an [example](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/scale.tsv).
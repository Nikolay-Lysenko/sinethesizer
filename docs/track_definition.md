# Guide on Track Definition

A track to be recorded as WAV file should be defined as a tab-separated file. Such file must have rows that represent sound events (loosely speaking, an event is a played note) and columns that represent properties of events.

Required columns are as follows:

Column | Description
:-----: | :---------:
timbre | Name of timbre
start_time | Time when event starts (in seconds)
duration | Duration of event (in seconds) including release stage
frequency | Frequency of sound (in Hz) or note (like A4); some timbres may ignore it
volume | Relative volume of the most loud piece of event
location | Position of sound source; a float between -1 and 1 where -1 stands for left channel only and 1 stands for right channel only
effects | List of effects in JSON; each record must have field "name" with supported effect name and, optionally, parameters of the effect; left this field blank if no effects are needed

For more intuitive explanation, look at an [example](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/docs/examples/scale.tsv).
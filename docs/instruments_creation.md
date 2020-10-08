# Guide on Creating Instruments

This tool is based on a representation of a sound produced by an instrument as a modified with effects sum of some sound waves. If so, to define an instrument means to define these sound waves (called partials) and these effects (and, also, a technical constant for amplitude normalization). The following table summarizes what are arguments that are needed to create an instrument:

Parameter | Description
:-------: | :---------:
partials | List of partials (see below how they are defined)
amplitude_scaling | Float number selected to prevent clipping by audio playing devices; if clipping happens, decrease it
effects | [Effects](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/effects/registry.py) (e.g., tremolo) that are always applied to sum of partials; this field is optional

Each partial requires these arguments:

Parameter | Description
:-------: | :---------:
wave | Definition of wave that forms the partial (see below for more details)
frequency_ratio | Ratio of this partial's frequency to fundamental frequency
amplitude_ratio | Declared ratio of this partial's peak amplitude to peak amplitude of the fundamental (both at maximum velocity); actual amplitude ratio may be different if effects applied to the partial and to the fundamental are not the same
event_to_amplitude_factor_fn | [Function](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/synth/event_to_amplitude_factor.py) that maps event to its multiplicative contribution to partial's amplitude; for example, this function can define dependence of amplitude on velocity
detuning_to_amplitude | Mapping from a detuning size (in semitones) to amplitude factor of a wave with the corresponding detuned frequency; this argument can be useful, because sum of slightly detuned waves sounds less artificial than one pure wave
random_detuning_range | Range of additional random detuning (in semitones); this argument can be useful, because if it is more than 0, a note played for the second time sounds not exactly like for the first time
effects | [Effects](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/effects/registry.py) (e.g., tremolo) that are always applied to this partial; this field is optional

Further, each wave has these parameters:

Parameter | Description
:-------: | :---------:
waveform | Form of wave; one of 'sine', 'square', 'triangle', 'sawtooth', and 'white_noise'
phase | Phase shift of a wave (in radians); this field is optional
amplitude_envelope_fn | [Function](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/envelopes/registry.py) that takes parameters such as duration, velocity, and frame rate as inputs and returns amplitude [envelope](https://en.wikipedia.org/wiki/Envelope_(music)) of output wave
modulator | Parameters of a wave that modulates frequency of original wave (see below); this field is optional

Finally, modulator is defined by these arguments:

Parameter | Description
:-------: | :---------:
waveform | Form of wave; one of 'sine', 'square', 'triangle', 'sawtooth', and 'white_noise'
frequency_ratio_numerator | Numerator in ratio of modulating wave frequency to that of a modulated wave
frequency_ratio_denominator | Denominator in ratio of modulating wave frequency to that of a modulated wave
phase | Phase shift of a wave (in radians); this field is optional
modulation_index_envelope_fn | [Function](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/envelopes/registry.py) that takes parameters such as duration, velocity, and frame rate as inputs and returns amplitude envelope of modulating wave

All listed above parameters must be set inside of a YAML file of particular structure. Look at an [example](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/presets/demo.yml) of such file to see how it can be done.

# Guide on Creating Instruments

This tool is based on a representation of a sound produced by an instrument as a modified with effects sum of some sound waves. If so, to define an instrument means to define these sound waves (called partials) and these effects (and, also, a technical constant for amplitude normalization). The following table summarizes what are arguments that are needed to create an instrument:

Parameter | Description | Required
:-------: | :---------: | :------:
partials | List of partials (see below how they are defined) | Yes
amplitude_scaling | Float number selected to prevent clipping by audio playing devices; if clipping happens, decrease it | Yes
effects | [Effects](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/effects/registry.py) (e.g., overdrive) that are always applied to sum of partials | No

Each partial requires these arguments:

Parameter | Description | Required
:-------: | :---------: | :------:
wave | Definition of wave that forms the partial (see below for more details) | Yes
frequency_ratio | Ratio of this partial's frequency to fundamental frequency | Yes
amplitude_ratio | Declared ratio of this partial's peak amplitude to peak amplitude of the fundamental (both at maximum velocity); actual amplitude ratio may be different if effects applied to the partial and to the fundamental are not the same | Yes
event_to_amplitude_factor_fn | [Function](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/synth/event_to_amplitude_factor.py) that maps event to its multiplicative contribution to partial's amplitude; for example, this function can define dependence of amplitude on velocity | Yes
random_detuning_range | Range of random detuning (in semitones); this argument can be useful, because if it is more than 0, a note played for the second time sounds not exactly like for the first time | No
detuning_to_amplitude | Mapping from additional detuning size (in semitones) to amplitude factor of a wave with the corresponding detuned frequency | No
effects | [Effects](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/effects/registry.py) (e.g., overdrive) that are always applied to this partial | No

Further, each wave has these parameters:

Parameter | Description | Required
:-------: | :---------: | :------:
waveform | Form of wave; one of 'sine', 'square', 'triangle', 'sawtooth', 'white_noise', 'pink_noise', and 'brown_noise' | Yes
amplitude_envelope_fn | [Function](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/envelopes/registry.py) that takes parameters such as duration, velocity, and frame rate as inputs and returns amplitude [envelope](https://en.wikipedia.org/wiki/Envelope_(music)) of output wave | Yes
phase | Phase shift of a wave (in radians) | No 
amplitude_modulator | Parameters of a wave that modulates amplitude of original wave (see below) | No
phase_modulator | Parameters of a wave that modulates phase of original wave (see below) | No
quasiperiodic_bandwidth | Bandwidth (in semitones) of instantaneous frequency random changes; these changes make output wave quasi-periodic and, hence, more natural| No

Finally, a modulator is defined by these arguments:

Parameter | Description | Required
:-------: | :---------: | :------:
waveform | Form of wave; one of 'sine', 'square', 'triangle', 'sawtooth', 'white_noise', 'pink_noise', and 'brown_noise' | Yes
frequency_ratio_numerator | Numerator in ratio of modulating wave frequency to that of a modulated wave | Yes
frequency_ratio_denominator | Denominator in ratio of modulating wave frequency to that of a modulated wave | Yes
modulation_index_envelope_fn | [Function](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/envelopes/registry.py) that takes parameters such as duration, velocity, and frame rate as inputs and returns amplitude envelope of modulating wave | Yes
phase | Phase shift of a wave (in radians) | No
use_ring_modulation | Boolean indicator whether to use ring modulation instead of classical amplitude modulation; this field affects nothing if it is set for phase modulator | No

All listed above parameters must be set inside of a YAML file of particular structure. Look at an [example](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/presets/demo.yml) of such file to see how it can be done.

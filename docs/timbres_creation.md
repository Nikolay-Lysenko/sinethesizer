# Guide on Creating Timbres

Additive synthesis means that a complicated wave with rich sounding can be composed of numerous simple waves. For a particular timbre, a wave with the lowest frequency is called the fundamental and other waves are called overtones. To create a new timbre, parameters of fundamental and its overtones must be set.

Currently, fundamental wave has two required parameters and one optional parameter:

Parameter | Description
:-------: | :---------:
fundamental_waveform | Form of wave; one of 'sine', 'square', 'triangle', and 'sawtooth'
fundamental_volume_envelope | Function that maps duration of sound event in seconds to [dynamic](https://en.wikipedia.org/wiki/Envelope_(music)) of fundamental's volume in time; supported names of [functions](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/synth/adsr_envelopes.py) are 'relative_adsr', 'absolute_adsr', 'trapezoid', and 'spike'
fundamental_effects | [Effects](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/synth/effects.py) (e.g., tremolo) that are always applied to fundamental; this field is optional

Each overtone has parameters similar to that of fundamental and some additional parameters:

Parameter | Description
:-------: | :---------:
waveform | Form of wave; one of 'sine', 'square', 'triangle', and 'sawtooth'
frequency_ratio | Ratio of frequency of this overtone to frequency of the fundamental; must be greater than 1
volume_share | Peak volume of this overtone divided by sum of peak volumes of all overtones and fundamental
volume_envelope | Function that maps duration of sound event in seconds to [dynamic](https://en.wikipedia.org/wiki/Envelope_(music)) of overtone's volume in time; supported names of [functions](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/synth/adsr_envelopes.py) are 'relative_adsr', 'absolute_adsr', 'trapezoid', and 'spike'
effects | [Effects](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/sinethesizer/synth/effects.py) (e.g., tremolo) that are always applied to this overtone; this field is optional

All listed above parameters must be set inside of a YAML file of particular structure. Look at an [example](https://github.com/Nikolay-Lysenko/sinethesizer/blob/master/presets/demo.yml) of such file to see how it can be done. Also this example shows that above parametrization is compatible not only with additive synthesis, but with subtractive synthesis and FM synthesis as well. 

"""
Define timbres of virtual instruments.

Author: Nikolay Lysenko
"""


from typing import List, NamedTuple

from sinethesizer.synth.adsr_envelopes import ENVELOPE_FN_TYPE
from sinethesizer.synth.effects import EFFECT_FN_TYPE


class OvertoneSpec(NamedTuple):
    """
    Specification of a particular overtone.

    :param waveform:
        form of wave;
        it can be one of 'sine', 'square', 'triangle', and 'sawtooth'
    :param frequency_ratio:
        ratio of frequency of the overtone to frequency of its fundamental
    :param volume_share:
        share of total volume (volume of all partials
        including the fundamental) that is taken by the overtone
        if all partials have unit volume on their envelopes
    :param volume_envelope_fn:
        function that maps duration in seconds and frame rate to
        volume envelope for this overtone
    :param effects:
        sound effects that are applied to the overtone
    """

    waveform: str
    frequency_ratio: float
    volume_share: float
    volume_envelope_fn: ENVELOPE_FN_TYPE
    effects: List[EFFECT_FN_TYPE]


class TimbreSpec(NamedTuple):
    """
    Specification of a particular timbre.

    :param fundamental_waveform:
        form of fundamental wave;
        it can be one of 'sine', 'square', 'triangle', and 'sawtooth'
    :param fundamental_volume_envelope_fn:
        function that maps duration in seconds and frame rate to
        volume envelope for the fundamental
    :param fundamental_effects:
        sound effects that are applied to the fundamental
    :param overtones_specs:
        list of specifications of overtones
    """

    fundamental_waveform: str
    fundamental_volume_envelope_fn: ENVELOPE_FN_TYPE
    fundamental_effects: List[EFFECT_FN_TYPE]
    overtones_specs: List[OvertoneSpec]

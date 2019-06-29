"""
Synthesize sound.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.synth.timbre import TimbreSpec
from sinethesizer.synth.waves import generate_wave
from sinethesizer.synth.utils import calculate_overtones_share


def synthesize(
        timbre_spec: TimbreSpec, frequency: float, volume: float,
        duration: float, location: float, max_channel_delay: float,
        frame_rate: int
) -> np.ndarray:
    """
    Synthesize sound fragment that corresponds to one note.

    :param timbre_spec:
        specification of a timbre
    :param frequency:
        frequency of fundamental in Hz
    :param volume:
        volume of the sound fragment
    :param duration:
        duration of fragment to be generated in seconds
    :param location:
        location of sound source;
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds;
        it is a measure of potential size of space occupied by sound sources
    :param frame_rate:
        number of frames per second
    :return:
        sound wave represented as timeline of pressure deviations
    """
    envelope = timbre_spec.fundamental_volume_envelope_fn(duration, frame_rate)
    overtones_share = calculate_overtones_share(timbre_spec)
    fundamental_share = 1 - overtones_share
    sound = generate_wave(
        timbre_spec.fundamental_waveform,
        frequency,
        volume * fundamental_share * envelope,
        location,
        max_channel_delay,
        frame_rate
    )
    for effect_fn in timbre_spec.fundamental_effects:
        sound = effect_fn(sound, frame_rate)
    for overtone_spec in timbre_spec.overtones_specs:
        envelope = overtone_spec.volume_envelope_fn(duration, frame_rate)
        overtone_sound = generate_wave(
            overtone_spec.waveform,
            overtone_spec.frequency_ratio * frequency,
            volume * overtone_spec.volume_share * envelope,
            location,
            max_channel_delay,
            frame_rate
        )
        for effect_fn in overtone_spec.effects:
            overtone_sound = effect_fn(overtone_sound, frame_rate)
        sound += overtone_sound
    return sound

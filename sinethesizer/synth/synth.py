"""
Synthesize sound.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.synth.timbre import TimbreSpec
from sinethesizer.utils.waves import generate_stereo_wave


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
        (without delays between channels and prolongations caused by effects)
    :param location:
        location of sound source;
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds (for Haas effect);
        it is a measure of potential size of space occupied by sound sources
    :param frame_rate:
        number of frames per second
    :return:
        sound wave represented as timeline of pressure deviations
    """
    envelope = timbre_spec.fundamental_volume_envelope_fn(duration, frame_rate)
    sound = generate_stereo_wave(
        timbre_spec.fundamental_waveform,
        frequency,
        envelope,
        frame_rate,
        location,
        max_channel_delay
    )
    sound_info = {'frame_rate': frame_rate, 'fundamental_frequency': frequency}
    for effect_fn in timbre_spec.fundamental_effects:
        sound = effect_fn(sound, sound_info)
    for overtone_spec in timbre_spec.overtones_specs:
        envelope = overtone_spec.volume_envelope_fn(duration, frame_rate)
        overtone_frequency = overtone_spec.frequency_ratio * frequency
        overtone_sound = generate_stereo_wave(
            overtone_spec.waveform,
            overtone_frequency,
            overtone_spec.volume_ratio * envelope,
            frame_rate,
            location,
            max_channel_delay,
            int(round((frame_rate / overtone_frequency * overtone_spec.phase)))
        )
        for effect_fn in overtone_spec.effects:
            overtone_sound = effect_fn(overtone_sound, sound_info)
        sound += overtone_sound
    sound *= volume / np.max(np.abs(sound))
    return sound

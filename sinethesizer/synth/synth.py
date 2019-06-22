"""
Synthesize sound.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.synth.timbres import TimbreSpec
from sinethesizer.synth.waves import generate_wave
from sinethesizer.synth.utils import calculate_overtones_share


def synthesize(
        timbre_spec: TimbreSpec,
        frequency: int, volume: float, duration: int, frame_rate: int = 44100
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
    :param frame_rate:
        number of frames per second
    :return:
        sound wave represented as timeline of pressure deviations
    """
    duration_in_frames = duration * frame_rate
    envelope = timbre_spec.fundamental_volume_envelope_fn(duration_in_frames)
    overtones_share = calculate_overtones_share(timbre_spec)
    fundamental_share = 1 - overtones_share
    sound = generate_wave(
        timbre_spec.fundamental_waveform,
        frequency,
        frame_rate,
        volume * fundamental_share * envelope
    )
    for overtone_spec in timbre_spec.overtones_specs:
        envelope = overtone_spec.volume_envelope_fn(duration_in_frames)
        overtone_sound = generate_wave(
            overtone_spec.waveform,
            overtone_spec.frequency_ratio * frequency,
            frame_rate,
            volume * overtone_spec.volume_share * envelope
        )
        sound += overtone_sound
    return sound

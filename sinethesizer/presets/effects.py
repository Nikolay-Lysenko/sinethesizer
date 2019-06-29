"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.synth.waves import generate_wave


def tremolo(
        sound: np.ndarray, frame_rate: int,
        frequency: float = 50, amplitude: float = 0.5
) -> np.ndarray:
    """
    Make sound volume vibrating.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param frequency:
        frequency of volume oscillations (in Hz)
    :param amplitude:
        relative amplitude of volume oscillation, must be between 0 and 1
    :return:
        sound with vibrating volume
    """
    amplitudes = amplitude * np.ones(sound.shape[1])
    volume_wave = generate_wave(
        'sine', frequency, amplitudes,
        location=0, max_channel_delay=0, frame_rate=frame_rate
    )
    volume_wave += 1
    sound *= volume_wave
    return sound

"""
Do some auxiliary tasks.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.synth.timbre import TimbreSpec
from sinethesizer.synth.waves import NAME_TO_WAVEFORM, generate_wave


def calculate_overtones_share(timbre_spec: TimbreSpec) -> float:
    """
    Calculate volume share of all overtones.

    :param timbre_spec:
        specification of a timbre
    :return:
        total volume share of all overtones if all partials have unit volume
        on their envelopes
    """
    overtones_share = sum(x.volume_share for x in timbre_spec.overtones_specs)
    overtones_share = overtones_share or 0
    return overtones_share


def oscillate_between_sounds(
        sounds: np.ndarray, frame_rate: int, frequency: float,
        waveform: str = 'sine'
) -> np.ndarray:
    """
    Combine multiple sounds into one sound by oscillating between them.

    :param sounds:
        array of shape (n_sounds, n_channels, n_frames)
    :param frame_rate:
        number of frames per second
    :param frequency:
        frequency of oscillations between sound sources
    :param waveform:
        form of oscillations wave
    :return:
        sound composed from input sounds
    """
    step = 2 / (sounds.shape[0] - 1)
    thresholds = np.arange(-1, 1 + 1e-7, step)
    weights = np.tile(thresholds.reshape((-1, 1)), (1, sounds.shape[2]))
    wave = generate_wave(
        waveform, frequency, np.ones(sounds.shape[2]), 0, 0, frame_rate
    )
    wave = wave[0, :]
    weights = (
        (1 - np.abs(weights - wave) / step) * (np.abs(weights - wave) < step)
    )
    weights = weights.reshape((weights.shape[0], 1, weights.shape[1]))
    sound = np.sum(sounds * weights, axis=0)
    return sound


def validate_timbre_spec(timbre_spec: TimbreSpec) -> None:
    """
    Validate specification of a timbre.

    :param timbre_spec:
        specification of a timbre
    :return:
        None
    """
    fundamental_waveform = timbre_spec.fundamental_waveform
    if fundamental_waveform not in NAME_TO_WAVEFORM.keys():
        raise ValueError(
            f"Unknown name of fundamental waveform: {fundamental_waveform}."
        )

    if len(timbre_spec.overtones_specs) == 0:
        return

    for overtone_spec in timbre_spec.overtones_specs:
        if overtone_spec.waveform not in NAME_TO_WAVEFORM.keys():
            raise ValueError(
                f"Unknown name of overtone waveform: {overtone_spec.waveform}."
            )

    min_ratio = min(x.frequency_ratio for x in timbre_spec.overtones_specs)
    if min_ratio <= 1:
        raise ValueError(
            "All overtones must have higher frequencies than the fundamental."
        )

    overtones_share = calculate_overtones_share(timbre_spec)
    if not (0 <= overtones_share < 1):
        raise ValueError(
            "Volume share of overtones must be inside [0, 1), "
            f"found: {overtones_share}."
        )

"""
Change power and amplitude distribution across frequencies.

Author: Nikolay Lysenko
"""


from typing import Dict, List, Union

import numpy as np
from scipy.signal import convolve, firwin2


def apply_equalizer(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        breakpoint_frequencies: List[float], gains: List[float],
        **kwargs
) -> np.ndarray:
    """
    Change power and amplitude distribution across frequencies.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param breakpoint_frequencies:
        frequencies (in Hz) that correspond to breaks in frequency response
        of equalizer
    :param gains:
        relative gains at corresponding breakpoint frequencies; a gain at an
        intermediate frequency is linearly interpolated
    :return:
        sound with altered frequency balance
    """
    nyquist_frequency = 0.5 * event.frame_rate
    breakpoint_frequencies = [
        x / nyquist_frequency for x in breakpoint_frequencies
    ]
    gains = [x for x in gains]  # Copy it to prevent modifying original list.
    if breakpoint_frequencies[0] != 0:
        breakpoint_frequencies.insert(0, 0)
        gains.insert(0, gains[0])
    if breakpoint_frequencies[-1] != 1:
        breakpoint_frequencies.append(1)
        gains.append(gains[-1])
    # `fir_size` is odd, because else there are constraints on `gains`.
    fir_size = 2 * int(round(event.frame_rate / 100)) + 1
    fir = firwin2(fir_size, breakpoint_frequencies, gains, **kwargs)
    sound = np.vstack((
        convolve(sound[0, :], fir, mode='same'),
        convolve(sound[1, :], fir, mode='same'),
    ))
    return sound


def apply_equalizer_envelope(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        envelope_points: List[Dict[str, Union[float, List[float]]]],
        **kwargs
) -> np.ndarray:
    """
    Change distribution across frequencies with equalizer that depends on time.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param envelope_points:
        points that define equalizer envelope; each point is a dictionary that
        has three keys: 'relative_position' - a float between 0 and 1 that
        defines position of the point on the envelope, 'breakpoint_frequencies'
        - settings of breakpoint frequencies of the equalizer applied at this
        point, and 'gains' - settings of gains of the equalizer applied at this
        point; at any other point, output is a linear interpolation
    :return:
        sound with dynamically altered frequency balance
    """
    n_frames = sound.shape[1]
    indices = []
    firs_params = []
    for envelope_point in envelope_points:
        index = int(round(envelope_point['relative_position'] * n_frames))
        indices.append(index)
        fir_params = {
            'breakpoint_frequencies': envelope_point['breakpoint_frequencies'],
            'gains': envelope_point['gains']
        }
        firs_params.append(fir_params)
    indices.insert(0, indices[0])
    indices.append(indices[-1])

    processed_sound = np.zeros_like(sound)
    zipped = zip(indices, indices[1:], indices[2:], firs_params)
    for start_index, center_index, end_index, fir_params in zipped:
        fragment = sound[:, start_index:end_index]
        processed_fragment = apply_equalizer(
            fragment, event, **fir_params, **kwargs
        )
        if center_index - start_index > 0:
            asc_weights = np.linspace(0, 1, center_index - start_index, False)
        else:
            asc_weights = np.array([])
        if end_index - center_index > 0:
            desc_weights = np.linspace(1, 0, end_index - center_index)
        else:
            desc_weights = np.array([])
        weights = np.hstack((asc_weights, desc_weights))

        processed_fragment *= weights
        processed_sound[:, start_index:end_index] += processed_fragment
    return processed_sound

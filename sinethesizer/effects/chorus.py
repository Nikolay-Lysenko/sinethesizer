"""
Imitate playing sounds by choirs or ensembles.

Author: Nikolay Lysenko
"""


from typing import Any

import numpy as np

from sinethesizer.effects.vibrato import apply_vibrato
from sinethesizer.utils.misc import sum_two_sounds


def apply_chorus(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        original_sound_gain: float, copies_params: list[dict[str, Any]]
) -> np.ndarray:
    """
    Apply chorus effect (or flanger effect if delay times are small enough).

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param original_sound_gain:
        amplitude gain for original non-delayed sound
    :param copies_params:
        list of dictionaries each of which contains:
        1) delay time for a current copy;
        2) amplitude gain for the copy;
        3) parameters of vibrato that should be applied to the copy
    :return:
        enriched sound somehow resembling sounds produced by choirs or ensembles
    """
    processed_copies = []
    for copy_params in copies_params:
        delay_in_sec = copy_params['delay']
        gain = copy_params['gain']
        vibrato_params = {k: v for k, v in copy_params.items() if k not in ['delay', 'gain']}
        detuned_copy = apply_vibrato(sound, event, **vibrato_params)
        detuned_copy *= gain
        n_frames_with_silence = int(round(delay_in_sec * event.frame_rate))
        silence = np.zeros((sound.shape[0], n_frames_with_silence))
        processed_copy = np.hstack((silence, detuned_copy))
        processed_copies.append(processed_copy)
    sound *= original_sound_gain
    for processed_copy in processed_copies:
        sound = sum_two_sounds(sound, processed_copy)
    return sound

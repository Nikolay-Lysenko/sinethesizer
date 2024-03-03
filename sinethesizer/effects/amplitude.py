"""
Control amplitude of sound (and hence its volume).

Author: Nikolay Lysenko
"""


from typing import Any

import numpy as np

from sinethesizer.envelopes import get_envelopes_registry


def apply_amplitude_normalization(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        value_at_max_velocity: float, quantile: float = 1,
        value_on_velocity_order: float = 1, value_at_zero_velocity: float = 0
) -> np.ndarray:
    """
    Normalize amplitude of sound.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param value_at_max_velocity:
        new value of specified quantile of absolute pressure deviations at maximum velocity
    :param quantile:
        quantile of absolute pressure deviations that is used for scaling
    :param value_on_velocity_order:
        coefficient that determines dependence of amplitude quantile value on velocity
    :param value_at_zero_velocity:
        new value of specified quantile of absolute pressure deviations at zero velocity
    :return:
        sound with amplitude normalized to new value
    """
    coef = event.velocity ** value_on_velocity_order
    diff = value_at_max_velocity - value_at_zero_velocity
    new_quantile_value = value_at_zero_velocity + coef * diff
    quantile_value = np.quantile(np.abs(sound), quantile)
    sound *= new_quantile_value / quantile_value
    return sound


def apply_compressor(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        threshold: float, quantile: float = 1, chunk_size_in_cycles: float = 3
) -> np.ndarray:
    """
    Limit maximum amplitude of the sound.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param threshold:
        ratio of maximum output amplitude to maximum possible amplitude that is not clipped by
        playing devices
    :param quantile:
        quantile of absolute pressure deviations that is used for scaling
    :param chunk_size_in_cycles:
        size of one window to be processed independently (in number of fundamental frequency's
        periods); the higher it is, the less probable artifacts are, but also the slower
        compressor reaction is
    :return:
        sound of limited amplitude
    """
    scaling_coefs = []
    previous_ratio = 1
    chunk_size_in_frames = chunk_size_in_cycles * event.frame_rate / event.frequency
    n_chunks = int(round(sound.shape[1] / chunk_size_in_frames))
    for chunk in np.array_split(sound, n_chunks, axis=1):
        current_value = np.quantile(np.abs(chunk), quantile)
        current_ratio = min(threshold / current_value, 1)
        current_scaling_coefs = np.linspace(previous_ratio, current_ratio, chunk.shape[1], False)
        scaling_coefs.append(current_scaling_coefs)
        previous_ratio = current_ratio
    scaling_coefs = np.hstack(scaling_coefs)
    sound *= scaling_coefs
    return sound


def apply_envelope_shaper(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        envelope_params: dict[str, Any], quantile: float = 1, chunk_size_in_cycles: float = 3,
        initial_rescaling_ratio: float = 0, forced_fading_ratio: float = 0
) -> np.ndarray:
    """
    Change envelope in order to make it closer to the specified envelope.

    In particular, this effect can be useful in the following situations:
    1) If noise is filtered and after that its original envelope is lost,
       this effect can restore it;
    2) To have uniform overdrive, the overdrive effect should be applied to
       sound with constant envelope and then this effect can be applied
       in order to set desired envelope.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param envelope_params:
        name of envelope generating function and its arguments
    :param quantile:
        quantile of absolute pressure deviations that is used for scaling
    :param chunk_size_in_cycles:
        size of one window to be processed independently (in number of fundamental frequency's
        periods); the higher it is, the less probable artifacts are, but also the higher deviations
        from the specified envelope are
    :param initial_rescaling_ratio:
        ratio for rescaling amplitude of the first frames;
        set it to 1 if sound already has proper attack;
        set it to 0 if attack is not set and smooth attack is needed
    :param forced_fading_ratio:
        ratio that defines number of last frames to which additional fading is applied;
        such fading prevents clipping and might be useful if `sound` has not smooth release
    :return:
        sound with new envelope
    """
    envelopes_registry = get_envelopes_registry()
    envelope_fn = envelopes_registry[envelope_params['name']]
    envelope = envelope_fn(event, **{k: v for k, v in envelope_params.items() if k != 'name'})
    if len(envelope) != sound.shape[1]:
        raise ValueError(
            "Only envelopes of the same length are supported, but "
            f"sound length is {sound.shape[1]} and envelope length is {len(envelope)}."
        )

    scaling_coefs = []
    previous_ratio = initial_rescaling_ratio
    chunk_size_in_frames = chunk_size_in_cycles * event.frame_rate / event.frequency
    n_chunks = int(round(sound.shape[1] / chunk_size_in_frames))
    split_sound = np.array_split(sound, n_chunks, axis=1)
    split_envelope = np.array_split(envelope, n_chunks, axis=0)
    for sound_chunk, envelope_chunk in zip(split_sound, split_envelope):
        current_value = np.quantile(np.abs(sound_chunk), quantile)
        current_ratio = np.mean(envelope_chunk) / current_value
        current_scaling_coefs = np.linspace(
            previous_ratio, current_ratio, sound_chunk.shape[1], False
        )
        scaling_coefs.append(current_scaling_coefs)
        previous_ratio = current_ratio
    scaling_coefs = np.hstack(scaling_coefs)

    if forced_fading_ratio > 0:
        forced_fading_duration_in_frames = int(round(forced_fading_ratio * sound.shape[1]))
        scaling_coefs *= np.hstack((
            np.ones(len(scaling_coefs) - forced_fading_duration_in_frames),
            np.linspace(1, 0, forced_fading_duration_in_frames, False)
        ))
    sound *= scaling_coefs
    return sound

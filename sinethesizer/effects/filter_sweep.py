"""
Apply sound effects based on filters with oscillating cutoffs.

This type of effects is a partial case of equalizer automation. If automation
based on a free-form envelope and not on an oscillator is needed, look at
`automation` module.

Author: Nikolay Lysenko
"""


from typing import List, Optional, Tuple

import numpy as np

from sinethesizer.effects.filter import apply_frequency_filter
from sinethesizer.oscillators import generate_mono_wave


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
    thresholds = np.linspace(-1, 1, sounds.shape[0])
    weights = np.tile(thresholds.reshape((-1, 1)), (1, sounds.shape[2]))
    wave = generate_mono_wave(
        waveform,
        frequency,
        np.ones(sounds.shape[2]),
        frame_rate
    )
    step = 2 / (sounds.shape[0] - 1)
    weights = (
        (1 - np.abs(weights - wave) / step) * (np.abs(weights - wave) < step)
    )
    weights = weights.reshape((weights.shape[0], 1, weights.shape[1]))
    sound = np.sum(sounds * weights, axis=0)
    return sound


def apply_filter_sweep(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        kind: str = 'absolute',
        bands: List[Tuple[Optional[float], Optional[float]]] = None,
        invert: bool = False, order: int = 25,
        frequency: float = 6, waveform: str = 'sine'
) -> np.ndarray:
    """
    Filter some frequencies with oscillating cutoffs.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param kind:
        if it is set to 'absolute', `bands` must be defined in Hz;
        if it is set to 'relative', `bands` must be defined as ratios to
        fundamental frequency
    :param bands:
        list of pairs of minimum and maximum cutoff frequencies;
        oscillations are between sounds obtained from input sound after
        applying filters with these cutoff frequencies
    :param invert:
        if it is `True`, for bands with both cutoff frequencies set not to
        `None`, band-stop filters are applied instead of band-pass filters
    :param order:
        order of filters; the higher it is, the steeper cutoffs are
    :param frequency:
        frequency of oscillations between filtered sounds (in Hz)
    :param waveform:
        form of wave that specifies oscillations between filtered sounds
    :return:
        sound filtered with varying cutoff frequencies
    """
    bands = bands or [(None, None)]
    if len(bands) == 1:
        sound = apply_frequency_filter(
            sound, event, kind,
            bands[0][0], bands[0][1], invert, order
        )
        return sound
    filtered_sounds = [
        apply_frequency_filter(
            sound, event, kind,
            min_cutoff_frequency, max_cutoff_frequency, invert, order
        )
        for min_cutoff_frequency, max_cutoff_frequency in bands
    ]
    filtered_sounds = [
        x.reshape((1, x.shape[0], x.shape[1])) for x in filtered_sounds
    ]
    filtered_sounds = np.concatenate(filtered_sounds)
    sound = oscillate_between_sounds(
        filtered_sounds, event.frame_rate, frequency, waveform
    )
    return sound


def apply_absolute_phaser(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        min_center: float = 220, max_center: float = 880,
        band_width: float = 20, n_bands: int = 10, order: int = 25,
        frequency: float = 5, waveform: str = 'sine',
        original_share: float = 0.75, wahwah: bool = False
) -> np.ndarray:
    """
    Apply phaser effect with border parameters defined in Hz.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param min_center:
        central frequency of the lowest band (in Hz)
    :param max_center:
        central frequency of the highest band (in Hz)
    :param band_width:
        width of sweeping band (in Hz)
    :param n_bands:
        number of band positions to consider; the higher it is, the more close
        to classical phaser result is, but also the longer computations are
        and the higher RAM consumption is during track creation
    :param order:
        order of filters; the higher it is, the steeper cutoffs are
    :param frequency:
        frequency of sweeping band oscillations;
        the higher it is, the more input sound is distorted
    :param waveform:
        form of wave of sweeping band oscillations
    :param original_share:
        share of original sound in resulting sound
    :param wahwah:
        if it is `True`, band-pass filters are used instead of band-stop
        filters and so the effect to be applied is called wah-wah, not phaser
    :return:
        phased sound
    """
    bands = [
        (center - band_width / 2, center + band_width / 2)
        for center in np.linspace(min_center, max_center, n_bands)
    ]
    invert = not wahwah
    filtered_sound = apply_filter_sweep(
        sound, event, 'absolute', bands, invert, order, frequency, waveform
    )
    sound = original_share * sound + (1 - original_share) * filtered_sound
    return sound


def apply_relative_phaser(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        min_center_ratio: float = 1.0, max_center_ratio: float = 4.0,
        relative_band_width: float = 0.1, n_bands: int = 10, order: int = 25,
        frequency: float = 5, waveform: str = 'sine',
        original_share: float = 0.75, wahwah: bool = False
) -> np.ndarray:
    """
    Apply phaser effect with border parameters defined as ratios.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param min_center_ratio:
        central frequency of the lowest band as ratio to fundamental frequency
    :param max_center_ratio:
        central frequency of the highest band as ratio to fundamental frequency
    :param relative_band_width:
        width of sweeping band as ratio to fundamental frequency
    :param n_bands:
        number of band positions to consider; the higher it is, the more close
        to classical phaser result is, but also the longer computations are
        and the higher RAM consumption is during track creation
    :param order:
        order of filters; the higher it is, the steeper cutoffs are
    :param frequency:
        frequency of sweeping band oscillations;
        the higher it is, the more input sound is distorted
    :param waveform:
        form of wave of sweeping band oscillations
    :param original_share:
        share of original sound in resulting sound
    :param wahwah:
        if it is `True`, band-pass filters are used instead of band-stop
        filters and so the effect to be applied is called wah-wah, not phaser
    :return:
        phased sound
    """
    fundamental_frequency = event.frequency
    min_center = min_center_ratio * fundamental_frequency
    max_center = max_center_ratio * fundamental_frequency
    band_width = relative_band_width * fundamental_frequency
    sound = apply_absolute_phaser(
        sound, event, min_center, max_center, band_width,
        n_bands, order, frequency, waveform, original_share, wahwah
    )
    return sound


def apply_phaser(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        kind: str = 'absolute', *args, **kwargs
) -> np.ndarray:
    """
    Apply phaser effect.

    Here, phaser is defined as weighted sum of:
    1) original sound;
    2) original sound modified by sweeping band-stop filter of narrow band.

    Note that playing with arguments can significantly change resulting sound
    and some settings produce awkward non-musical sounds. Also note that this
    effect should be applied only to sounds with rich spectrum.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param kind:
        kind of phaser; supported values are 'absolute' and 'relative'
    :return:
        phased sound
    """
    if kind == 'absolute':
        sound = apply_absolute_phaser(sound, event, *args, **kwargs)
    elif kind == 'relative':
        sound = apply_relative_phaser(sound, event, *args, **kwargs)
    else:
        raise ValueError(
            f"Kind must be either 'absolute' or 'relative', but found: {kind}"
        )
    return sound

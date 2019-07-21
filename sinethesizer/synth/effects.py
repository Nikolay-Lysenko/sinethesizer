"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt

from sinethesizer.synth.waves import generate_wave


EFFECT_FN_TYPE = Callable[[np.ndarray, int],  np.ndarray]


def frequency_filter(
        sound: np.ndarray, frame_rate: int,
        min_frequency: Optional[float] = None,
        max_frequency: Optional[float] = None,
        invert: bool = False, order: int = 10
) -> np.ndarray:
    """
    Filter some frequencies from original sound.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param min_frequency:
        cutoff frequency for high-pass filtering (in Hz);
        there is no high-pass filtering by default
    :param max_frequency:
        cutoff frequency for low-pass filtering (in Hz);
        there is no low-pass filtering by default
    :param invert:
        if it is `True` and both `min_frequency` and `max_frequency`
        are passed, band-stop filter is applied instead of band-pass filter
    :param order:
        order of the filter; the higher it is, the steeper cutoff is
    :return:
        sound with some frequencies muted
    """
    invert = invert and min_frequency is not None and max_frequency is not None
    filter_type = 'bandstop' if invert else 'bandpass'
    nyquist_frequency = 0.5 * frame_rate
    min_frequency = min_frequency or 1e-2  # Arbitrary small positive number.
    max_frequency = max_frequency or nyquist_frequency - 1e-2
    min_threshold = min_frequency / nyquist_frequency
    max_threshold = max_frequency / nyquist_frequency
    second_order_sections = butter(
        order, [min_threshold, max_threshold], btype=filter_type, output='sos'
    )  # 'ba' is not used, because sometimes it lacks numerical stability.
    sound = sosfilt(second_order_sections, sound)
    return sound


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
        waveform,
        frequency,
        np.ones(sounds.shape[2]),
        frame_rate
    )
    wave = wave[0, :]
    weights = (
        (1 - np.abs(weights - wave) / step) * (np.abs(weights - wave) < step)
    )
    weights = weights.reshape((weights.shape[0], 1, weights.shape[1]))
    sound = np.sum(sounds * weights, axis=0)
    return sound


def filter_sweep(
        sound: np.ndarray, frame_rate: int,
        bands: List[Tuple[Optional[float], Optional[float]]] = None,
        invert: bool = False, order: int = 10,
        frequency: float = 6, waveform: str = 'sine'
) -> np.ndarray:
    """
    Filter some frequencies from sound with oscillating cutoffs.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param bands:
        list of pairs of minimum and maximum cutoff frequencies (in Hz);
        oscillations are between sounds obtained from input sound after
        applying filters with such cutoff frequencies
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
        sound = frequency_filter(
            sound, frame_rate, bands[0][0], bands[0][1], invert, order
        )
        return sound
    filtered_sounds = [
        frequency_filter(
            sound, frame_rate, min_cutoff_frequency, max_cutoff_frequency,
            invert, order
        )
        for min_cutoff_frequency, max_cutoff_frequency in bands
    ]
    filtered_sounds = [
        x.reshape((1, x.shape[0], x.shape[1])) for x in filtered_sounds
    ]
    filtered_sounds = np.concatenate(filtered_sounds)
    sound = oscillate_between_sounds(
        filtered_sounds, frame_rate, frequency, waveform
    )
    return sound


def overdrive(
        sound: np.ndarray, frame_rate: int,
        fraction_to_clip: float = 0.1, strength: float = 0.3
) -> np.ndarray:
    """
    Overdrive the sound.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param fraction_to_clip:
        fraction of the most outlying frames to be hard clipped
    :param strength:
        relative strength of distortion, must be between 0 and 1
    :return:
        overdriven sound
    """
    if not (0 < fraction_to_clip < 1):
        raise ValueError("Fraction to clip must be between 0 and 1.")
    if not (0 <= strength < 1):
        raise ValueError("Overdrive strength must be between 0 and 1.")
    _ = frame_rate  # All effects must have `frame_rate` argument.

    abs_sound = np.abs(sound)
    clipping_threshold = np.quantile(abs_sound, 1 - fraction_to_clip, axis=1)
    clipping_threshold = clipping_threshold.reshape((-1, 1))
    clipping_cond = abs_sound >= clipping_threshold
    distorted_sound = sound - strength * sound**3 / clipping_threshold**2
    clipped_sound = np.sign(sound) * (1 - strength) * clipping_threshold
    sound = (
        ~clipping_cond * distorted_sound
        + clipping_cond * clipped_sound
    )
    sound /= (1 - strength)
    return sound


def phaser(
        sound: np.ndarray, frame_rate: int,
        min_center: float = 220, max_center: float = 880,
        band_width: float = 20, n_bands: int = 10, order: int = 10,
        frequency: float = 5, waveform: str = 'sine',
        original_share: float = 0.75, wahwah: bool = False
) -> np.ndarray:
    """
    Apply phaser effect to sound.

    Here, phaser is defined as weighted sum of:
    1) original sound;
    2) original sound modified by sweeping band-stop filter of narrow band.

    Note that playing with arguments can significantly change resulting sound
    and some settings produce awkward non-musical sounds. Also note that this
    effect should be applied only to sounds with rich spectrum.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
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
    step = (max_center - min_center) / n_bands
    bands = [
        (center - band_width / 2, center + band_width / 2)
        for center in np.arange(min_center, max_center + 1e-7, step)
    ]
    invert = not wahwah
    filtered_sound = filter_sweep(
        sound, frame_rate, bands, invert, order, frequency, waveform
    )
    sound = original_share * sound + (1 - original_share) * filtered_sound
    return sound


def tremolo(
        sound: np.ndarray, frame_rate: int,
        frequency: float = 6, amplitude: float = 0.5, waveform: str = 'sine'
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
        relative amplitude of volume oscillations, must be between 0 and 1
    :param waveform:
        form of volume oscillations wave
    :return:
        sound with vibrating volume
    """
    if not (0 < amplitude <= 1):
        raise ValueError("Amplitude for tremolo must be between 0 and 1.")
    amplitudes = amplitude * np.ones(sound.shape[1])
    volume_wave = generate_wave(waveform, frequency, amplitudes, frame_rate)
    volume_wave += 1
    sound *= volume_wave
    return sound


def vibrato(
        sound: np.ndarray, frame_rate: int,
        frequency: float = 4, width: float = 0.2, waveform: str = 'sine'
) -> np.ndarray:
    """
    Make sound frequency vibrating.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param frequency:
        frequency of sound's frequency oscillations (in Hz)
    :param width:
        difference between the highest frequency of oscillating sound
        and the lowest frequency of oscillating sound (in semitones)
    :param waveform:
        form of frequency oscillations wave
    :return:
        sound with vibrating frequency
    """
    semitone = 2 ** (1 / 12)
    highest_to_lowest_ratio = semitone ** width
    # If x = 0, d(x + m * sin(2 * \pi * f * x))/dx = 1 + 2 * \pi * f * m.
    # If x = \pi, d(x + m * sin(2 * \pi * f * x))/dx = 1 - 2 * \pi * f * m.
    # Ratio of above right sides is `highest_to_lowest_ratio`.
    # Let us solve it for `m` (`max_delay`).
    max_delay = (
        (highest_to_lowest_ratio - 1)
        / ((highest_to_lowest_ratio + 1) * 2 * np.pi * frequency)
    )

    amplitudes = max_delay * frame_rate * np.ones(sound.shape[1])
    frequency_wave = generate_wave(waveform, frequency, amplitudes, frame_rate)
    time_indices = np.ones(sound.shape[1]).cumsum() - 1 + frequency_wave[0, :]

    upper_indices = np.ceil(time_indices).astype(int)
    upper_indices = np.clip(upper_indices, 0, sound.shape[1] - 1)
    upper_sound = sound[:, upper_indices]

    lower_indices = np.floor(time_indices).astype(int)
    lower_indices = np.clip(lower_indices, 0, sound.shape[1] - 1)
    lower_sound = sound[:, lower_indices]

    weights = time_indices - lower_indices
    sound = weights * upper_sound + (1 - weights) * lower_sound
    return sound


def get_effects_registry() -> Dict[str, EFFECT_FN_TYPE]:
    """
    Get mapping from effect names to functions that apply effects.

    :return:
        registry of effects
    """
    registry = {
        'filter': frequency_filter,
        'filter_sweep': filter_sweep,
        'overdrive': overdrive,
        'phaser': phaser,
        'tremolo': tremolo,
        'vibrato': vibrato
    }
    return registry

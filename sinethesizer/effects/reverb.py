"""
Imitate acoustic reflections within a room.

Author: Nikolay Lysenko
"""


import random
import warnings
from math import ceil
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import convolve


def apply_reverb(
        sound: np.ndarray, sound_info: Dict[str, Any],
        first_reflection_delay: float = 0.01,
        decay_duration: float = 0.09,
        amplitude_random_range: float = 0.1,
        n_early_reflections: int = 4,
        early_reflections_delay: float = 0.005,
        diffusion_delay_factor: float = 0.95,
        diffusion_delay_random_range: float = 0.1,
        original_sound_gain: float = 0.8,
        reverberations_gain: float = 0.6,
        keep_peak_volume: bool = True,
        random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Imitate acoustic reflections within a room.

    :param sound:
        sound to be modified
    :param sound_info:
        information about `sound` variable such as number of frames per second
        and its fundamental frequency (if it exists)
    :param first_reflection_delay:
        time (in seconds) between original sound and its first reflection
    :param decay_duration:
        total time (in seconds) since the first reflection until the last one
    :param amplitude_random_range:
        relative range where amplitude of a reflection can randomly vary
    :param n_early_reflections:
        number of equally spaced reflections
    :param early_reflections_delay:
        time (in seconds) between successive early reflections
    :param diffusion_delay_factor:
        exponential decay factor for time between successive late reflections
    :param diffusion_delay_random_range:
        relative range where time between successive late reflections
        can randomly vary
    :param original_sound_gain:
        fraction of amplitude of original sound that is kept
        in resulting sound
    :param reverberations_gain:
        fraction of amplitude of reverberations sum that is kept
        in resulting sound
    :param keep_peak_volume:
        if it is set to `True`, processed sound is rescaled to maintain its
        original peak volume which is usually changed due to wave interference
    :param random_seed:
        seed for pseudo-random number generator
    :return:
        reverberated sound
    """
    random.seed(random_seed)
    early_span = (n_early_reflections - 1) * early_reflections_delay
    late_reflections_duration = decay_duration - early_span
    if late_reflections_duration < 0:
        raise ValueError("Early reflections last more than all reflections.")
    max_late_reflections_duration = (
        # Right multiplier is sum of geometric progression: a + a^2 + a^3 + ...
        early_reflections_delay * (1 / (1 - diffusion_delay_factor) - 1)
    )
    padding = 0.01
    if max_late_reflections_duration < late_reflections_duration + padding:
        raise ValueError(
            "Inconsistent parameters. "
            "Increase `diffusion_delay_factor` or decrease `decay_duration`."
        )
    if diffusion_delay_factor > 1:  # pragma: no cover
        warnings.warn(
            "If `diffusion_delay_factor` > 1, reverb is not realistic.",
            UserWarning
        )

    frame_rate = sound_info['frame_rate']
    ir_duration_in_seconds = first_reflection_delay + decay_duration
    ir_duration_in_frames = ceil(frame_rate * ir_duration_in_seconds)
    impulse_response = np.zeros(ir_duration_in_frames)
    impulse_response[0] = original_sound_gain

    reflection_times = [first_reflection_delay]
    for i in range(n_early_reflections - 1):
        next_reflection_time = reflection_times[-1] + early_reflections_delay
        reflection_times.append(next_reflection_time)
    delay_without_randomness = early_reflections_delay
    while True:
        delay_without_randomness *= diffusion_delay_factor
        random_number = random.uniform(
            -diffusion_delay_random_range, diffusion_delay_random_range
        )
        delay = delay_without_randomness * (1 + random_number)
        reflection_time = reflection_times[-1] + delay
        if reflection_time < ir_duration_in_seconds:
            reflection_times.append(reflection_time)
        else:
            break

    for reflection_time in reflection_times:
        index = min(
            int(round(frame_rate * reflection_time)),
            ir_duration_in_frames - 1
        )
        amplitude = (ir_duration_in_seconds - reflection_time) / decay_duration
        amplitude *= reverberations_gain
        random_factor = 1 - random.uniform(-amplitude_random_range, 0)
        amplitude *= random_factor
        impulse_response[index] = amplitude

    original_peak_volume = np.max(np.abs(sound))
    sound = np.vstack((
        convolve(sound[0, :], impulse_response),
        convolve(sound[1, :], impulse_response)
    ))
    if keep_peak_volume:
        sound = original_peak_volume / np.max(np.abs(sound)) * sound
    return sound

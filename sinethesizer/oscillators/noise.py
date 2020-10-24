"""
Generate noise.

Author: Nikolay Lysenko
"""


from math import floor, log

import numpy as np
import scipy.signal


def generate_power_law_noise(
        duration_in_frames: int, frame_rate: int, psd_decay_order: float,
        exponential_step: float = 2
) -> np.ndarray:
    """
    Generate noise with bandwidth intensity decaying as power of frequency.

    :param duration_in_frames:
        number of frames with noise to be generated
    :param frame_rate:
        number of frames per second
    :param psd_decay_order:
        order of intensity (i.e., power, not amplitude) decay with frequency
    :param exponential_step:
        exponential step for defining filter parameters
    :return:
        noise
    """
    white_noise = np.random.normal(0, 0.3, duration_in_frames)
    if psd_decay_order == 0:
        return white_noise

    nyquist_frequency = frame_rate / 2
    audibility_threshold_in_hz = 20
    ratio = audibility_threshold_in_hz / nyquist_frequency
    inverse_ratio = 1 / ratio
    n_full_steps = floor(log(inverse_ratio, exponential_step))

    breakpoint_frequencies = np.logspace(
        0, n_full_steps, n_full_steps + 1,
        base=exponential_step
    )
    breakpoint_frequencies *= ratio
    breakpoint_frequencies = np.insert(breakpoint_frequencies, 0, 0)
    breakpoint_frequencies = np.append(breakpoint_frequencies, 1)

    gains = np.logspace(
        0, 0.5 * psd_decay_order * n_full_steps, n_full_steps + 1,
        base=1/exponential_step
    )
    gains = np.insert(gains, 0, 1)
    gains = np.append(gains, 0)  # It prevents aliasing.

    # Below constants are chosen empirically for pink and brown noises only.
    scaling_dict = {1.0: 11, 2.0: 29}
    scaling = scaling_dict.get(psd_decay_order, 1)
    gains *= scaling

    fir_size = 2 * int(round(frame_rate / 100)) + 1
    fir = scipy.signal.firwin2(fir_size, breakpoint_frequencies, gains)
    result = scipy.signal.convolve(white_noise, fir, mode='same')
    return result

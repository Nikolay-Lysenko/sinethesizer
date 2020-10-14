"""
Generate noise.

Author: Nikolay Lysenko
"""


import numpy as np
from scipy.signal import convolve, firwin2


def generate_power_law_noise(
        xs: np.ndarray, frame_rate: int, psd_decay_order: float,
        n_equalizer_points: int = 100
) -> np.ndarray:
    """
    Generate noise with bandwidth intensity decaying as power of frequency.

    :param xs:
        arrays of input data points; only its length is used
    :param frame_rate:
        number of frames per second in `xs`; it is used for computing
        filter size
    :param psd_decay_order:
        power of frequency in intensity denominator
    :param n_equalizer_points:
        number of points to approximate gain at each frequency
    :return:
        noise
    """
    white_noise = np.random.normal(0, 1, xs.shape)
    if psd_decay_order == 0:
        return white_noise
    breakpoint_frequencies = np.linspace(0, 1, n_equalizer_points)
    gains = (1 - breakpoint_frequencies) ** psd_decay_order
    fir_size = 2 * int(round(frame_rate / 100)) + 1
    fir = firwin2(fir_size, breakpoint_frequencies, gains)
    result = convolve(white_noise, fir, mode='same')
    return result

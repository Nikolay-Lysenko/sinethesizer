"""
Generate noise.

Author: Nikolay Lysenko
"""


import numpy as np
from scipy.signal import convolve, firwin2


def generate_power_law_noise(
        xs: np.ndarray, frame_rate: int, psd_decay_order: float,
        n_equalizer_points: int = 300
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
    white_noise = np.random.normal(0, 0.3, xs.shape)
    if psd_decay_order == 0:
        return white_noise
    nyquist_frequency = frame_rate / 2
    audibility_threshold_in_hz = 20
    ratio = audibility_threshold_in_hz / nyquist_frequency
    breakpoint_frequencies = np.linspace(ratio, 1, n_equalizer_points)
    gains = 1 / breakpoint_frequencies ** psd_decay_order
    breakpoint_frequencies = np.hstack((np.array([0]), breakpoint_frequencies))
    # Below constant is chosen empirically for pink and brown noise.
    # An effect named amplitude normalization can be used for finer control.
    gain_factor = 25
    gains = gain_factor * np.hstack((gains[:1], gains)) / gains[0]
    fir_size = 2 * int(round(frame_rate / 100)) + 1
    fir = firwin2(fir_size, breakpoint_frequencies, gains)
    result = convolve(white_noise, fir, mode='same')
    return result

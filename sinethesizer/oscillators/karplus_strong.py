"""
Generate similar to plucked strings sounds with Karplus-Strong method.

Author: Nikolay Lysenko
"""


from math import ceil

import numpy as np


def generate_karplus_strong_wave(
        frequency: float, duration_in_frames: int, frame_rate: int
) -> np.ndarray:
    """
    Generate wave with Karplus-Strong method.

    :param frequency:
        frequency of wave (in Hz)
    :param duration_in_frames:
        duration of output sound in frames
    :param frame_rate:
        number of frames per second
    :return:
        sound resembling a sound of a plucked string
    """
    block_size = int(round(frame_rate / frequency))
    block = np.ones(block_size)
    random_indices = np.random.choice(block_size, block_size // 2, False)
    block[random_indices] = -1

    results = []
    n_buffer_repetitions = ceil(duration_in_frames / block_size)
    for _ in range(n_buffer_repetitions):
        new_block = 0.5 * (block + np.append(block[1:], 0))
        new_block[-1] += 0.5 * new_block[0]
        results.append(new_block)
        block = new_block

    results = np.hstack(results)[:duration_in_frames]
    return results

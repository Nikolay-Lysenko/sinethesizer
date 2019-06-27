"""
Convert an array with pressure deviations timeline to WAV file.

Author: Nikolay Lysenko
"""


import numpy as np
import scipy.io.wavfile


def write_timeline_to_wav(
        output_path: str, timeline: np.ndarray, frame_rate: int
) -> None:
    """
    Write pressure deviations timeline to WAV file.

    :param output_path:
        path to resulting file
    :param timeline:
        sound represented as pressure deviations timeline
    :param frame_rate:
        number of frames per second
    :return:
        None
    """
    scipy.io.wavfile.write(output_path, frame_rate, timeline.T)

"""
Test `sinethesizer.io.numpy_to_wav` module.

Author: Nikolay Lysenko
"""


import pytest
import numpy as np

from sinethesizer.io.numpy_to_wav import write_timeline_to_wav


@pytest.mark.parametrize(
    "timeline, frame_rate",
    [(np.array([[1, 2, 3], [2, 3, 4]]), 10)]
)
def test_write_timeline_to_wav(
        path_to_tmp_file: str, timeline: np.ndarray, frame_rate: int
) -> None:
    """Test `write_timeline_to_wav` function."""
    write_timeline_to_wav(path_to_tmp_file, timeline, frame_rate)

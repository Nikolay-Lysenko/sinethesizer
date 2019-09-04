"""
Test sinethesizer.io.piano_roll_to_tsv` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest
import numpy as np

from sinethesizer.io.piano_roll_to_tsv import write_roll_to_tsv_file


@pytest.mark.parametrize(
    "roll, lowest_note, expected",
    [
        (
            np.array([
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 0],
            ]),
            'C4',
            [
                "timbre\tstart_time\tduration\tfrequency\tvolume\tlocation\teffects",
                "sine\t0\t3\tC#4\t1\t0\t",
                "sine\t0\t1\tD4\t1\t0\t",
                "sine\t2\t1\tC4\t1\t0\t",
            ]
        ),
        (
            np.array([
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [0, 0, 1, 0],
            ]),
            'C4',
            [
                "timbre\tstart_time\tduration\tfrequency\tvolume\tlocation\teffects",
                "sine\t0\t1\tC#4\t1\t0\t",
                "sine\t0\t1\tD4\t1\t0\t",
                "sine\t2\t1\tC4\t1\t0\t",
                "sine\t2\t1\tC#4\t1\t0\t",
                "sine\t3\t1\tD4\t1\t0\t",
            ]
        ),
        (
            np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 0],
            ]),
            'G4',
            [
                "timbre\tstart_time\tduration\tfrequency\tvolume\tlocation\teffects",
                "sine\t0\t1\tG4\t1\t0\t",
                "sine\t0\t1\tA4\t1\t0\t",
                "sine\t2\t1\tG4\t1\t0\t",
                "sine\t2\t2\tG#4\t1\t0\t",
            ]
        ),
    ]
)
def test_write_roll_to_tsv_file(
        path_to_tmp_file: str, roll: np.ndarray, lowest_note: str,
        expected: List[str]
) -> None:
    """Test `write_roll_to_tsv_file` function."""
    write_roll_to_tsv_file(
        roll, path_to_tmp_file, lowest_note,
        timbre='sine', step_in_seconds=1, volume=1
    )
    with open(path_to_tmp_file) as in_file:
        for expected_line in expected:
            result_line = in_file.readline().rstrip('\n')
            assert expected_line == result_line

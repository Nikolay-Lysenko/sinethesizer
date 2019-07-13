"""
Test sinethesizer.io.tsv_to_numpy` module.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import Any, Dict, List

import pytest
import numpy as np

from sinethesizer.io.tsv_to_numpy import convert_tsv_to_timeline
from sinethesizer.synth.adsr_envelopes import constant_with_linear_ends
from sinethesizer.synth.timbre import TimbreSpec


@pytest.mark.parametrize(
    "tsv_content, settings, expected",
    [
        (
            [
                "timbre\tstart_time\tduration\tfrequency\tvolume\tlocation\teffects",
                "sine\t1\t1\tA0\t1\t0\t",
                'sine\t2\t1\t1\t1\t0\t[{"name": "tremolo", "frequency": 1}]'
            ],
            {
                'frame_rate': 4,
                'trailing_silence': 1,
                'max_channel_delay': 0.02,
                'timbres_registry': {
                    'sine': TimbreSpec(
                        fundamental_waveform='sine',
                        fundamental_volume_envelope_fn=partial(
                            constant_with_linear_ends,
                            begin_share=0, end_share=0
                        ),
                        fundamental_effects=[],
                        overtones_specs=[]
                    )
                }
            },
            np.array([
                [
                    0, 0, 0, 0,
                    0, -0.707106781, -1, -0.707106781,
                    0, 1.5, 0, -0.5,
                    0, 0, 0, 0
                ],
                [
                    0, 0, 0, 0,
                    0, -0.707106781, -1, -0.707106781,
                    0, 1.5, 0, -0.5,
                    0, 0, 0, 0
                ]
            ])
        )
    ]
)
def test_convert_tsv_to_timeline(
        path_to_tmp_file: str, tsv_content: List[str],
        settings: Dict[str, Any], expected: np.ndarray
) -> None:
    """Test `convert_tsv_to_timeline` function."""
    with open(path_to_tmp_file, 'w') as tmp_tsv_file:
        for line in tsv_content:
            tmp_tsv_file.write(line + '\n')
    result = convert_tsv_to_timeline(path_to_tmp_file, settings)
    np.testing.assert_almost_equal(result, expected)

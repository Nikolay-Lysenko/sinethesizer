"""
Test `sinethesizer.io.tsv_to_events` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import pytest

from sinethesizer.io.tsv_to_events import convert_tsv_to_events
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "tsv_content, settings, expected",
    [
        (
            [
                "instrument\tstart_time\tduration\tfrequency\tvelocity\teffects",
                "sine\t1\t1\tA0\t1\t",
                'sine\t2\t1\t1\t1\t[{"name": "tremolo", "frequency": 1}]'
            ],
            {
                'frame_rate': 4,
                'trailing_silence': 1,
            },
            [
                Event(
                    instrument='sine',
                    start_time=1.0,
                    duration=1.0,
                    frequency=27.5,
                    velocity=1.0,
                    effects='',
                    frame_rate=4
                ),
                Event(
                    instrument='sine',
                    start_time=2.0,
                    duration=1.0,
                    frequency=1.0,
                    velocity=1.0,
                    effects='[{"name": "tremolo", "frequency": 1}]',
                    frame_rate=4
                ),
            ]
        ),
        (
            [
                "instrument\tstart_time\tduration\tfrequency\tvelocity\teffects\textra_column",
                "sine\t1\t1\tA0\t1\t\tsomething",
                'sine\t2\t1\t1\t1\t[{"name": "tremolo", "frequency": 1}]\tsomething'
            ],
            {
                'frame_rate': 4,
                'trailing_silence': 1,
            },
            [
                Event(
                    instrument='sine',
                    start_time=1.0,
                    duration=1.0,
                    frequency=27.5,
                    velocity=1.0,
                    effects='',
                    frame_rate=4
                ),
                Event(
                    instrument='sine',
                    start_time=2.0,
                    duration=1.0,
                    frequency=1.0,
                    velocity=1.0,
                    effects='[{"name": "tremolo", "frequency": 1}]',
                    frame_rate=4
                ),
            ]
        ),
    ]
)
def test_convert_tsv_to_timeline(
        path_to_tmp_file: str, tsv_content: List[str],
        settings: Dict[str, Any], expected: List[Event]
) -> None:
    """Test `convert_tsv_to_timeline` function."""
    with open(path_to_tmp_file, 'w') as tmp_tsv_file:
        for line in tsv_content:
            tmp_tsv_file.write(line + '\n')
    result = convert_tsv_to_events(path_to_tmp_file, settings)
    assert result == expected

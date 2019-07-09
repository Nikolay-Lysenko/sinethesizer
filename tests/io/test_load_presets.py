"""
Test `sinethesizer.io.load_presets` module.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import List, Dict

import numpy as np
import pytest

from sinethesizer.io.load_presets import create_timbres_registry
from sinethesizer.synth import synthesize
from sinethesizer.synth.timbre import TimbreSpec
from sinethesizer.synth.adsr_envelopes import constant_with_linear_ends


@pytest.mark.parametrize(
    "yaml_content, expected",
    [
        (
            [
                "---",
                "- name: sine",
                "  fundamental_waveform: sine",
                "  fundamental_volume_envelope:",
                "    name: constant_with_linear_ends"
            ],
            {
                'sine': TimbreSpec(
                    fundamental_waveform='sine',
                    fundamental_volume_envelope_fn=constant_with_linear_ends,
                    fundamental_effects=[],
                    overtones_specs=[]
                )
            }
        )
    ]
)
def test_create_timbres_registry(
        path_to_tmp_file: str, yaml_content: List[str],
        expected: Dict[str, TimbreSpec]
) -> None:
    """Test `create_timbres_registry` function."""
    with open(path_to_tmp_file, 'w') as tmp_yml_file:
        for line in yaml_content:
            tmp_yml_file.write(line + '\n')
    result = create_timbres_registry(path_to_tmp_file)

    play_note = partial(
        synthesize, frequency=440, volume=1, duration=1,
        location=0, max_channel_delay=0, frame_rate=8000
    )

    for name, timbre_spec in expected.items():
        resulting_sound = play_note(result[name])
        expected_sound = play_note(timbre_spec)
        np.testing.assert_allclose(resulting_sound, expected_sound)

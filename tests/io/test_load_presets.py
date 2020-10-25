"""
Test `sinethesizer.io.load_presets` module.

Author: Nikolay Lysenko
"""


import functools
from typing import List, Dict

import numpy as np
import pytest

from sinethesizer.effects.filter import apply_frequency_filter
from sinethesizer.io.load_presets import create_instruments_registry
from sinethesizer.synth import synthesize
from sinethesizer.synth.core import (
    Event, Instrument, ModulatedWave, Modulator, Partial
)
from sinethesizer.synth.event_to_amplitude_factor import (
    compute_amplitude_factor_as_power_of_velocity
)
from sinethesizer.envelopes.ahdsr import (
    create_generic_ahdsr_envelope,
    create_trapezoid_envelope
)


@pytest.mark.parametrize(
    "yaml_content, expected",
    [
        (
            [
                "---",
                "- name: sine",
                "  partials:",
                "    - wave:",
                "        waveform: sine",
                "        amplitude_envelope_fn:",
                "          name: trapezoid",
                "      frequency_ratio: 1.0",
                "      amplitude_ratio: 1.0",
                "      detuning_to_amplitude:",
                "        0.0: 1.0",
                "      random_detuning_range: 0.0",
                "  amplitude_scaling: 1.0",
            ],
            {
                'sine': Instrument(
                    partials=[
                        Partial(
                            wave=ModulatedWave(
                                waveform='sine',
                                amplitude_envelope_fn=create_trapezoid_envelope,
                                phase=0,
                                amplitude_modulator=None,
                                phase_modulator=None,
                                quasiperiodic_bandwidth=0
                            ),
                            frequency_ratio=1.0,
                            amplitude_ratio=1.0,
                            event_to_amplitude_factor_fn=functools.partial(
                                compute_amplitude_factor_as_power_of_velocity,
                                power=1
                            ),
                            detuning_to_amplitude={0.0: 1.0},
                            random_detuning_range=0.0,
                            effects=[]
                        )
                    ],
                    amplitude_scaling=1.0,
                    effects=[]
                )
            }
        ),
        (
            [
                "---",
                "- name: sine",
                "  partials:",
                "    - wave:",
                "        waveform: sine",
                "        amplitude_envelope_fn:",
                "          name: trapezoid",
                "        amplitude_modulator:",
                "          waveform: sine",
                "          frequency_ratio_numerator: 99",
                "          frequency_ratio_denominator: 100",
                "          modulation_index_envelope_fn:",
                "            name: trapezoid",
                "          use_ring_modulation: true",
                "      frequency_ratio: 1.0",
                "      amplitude_ratio: 1.0",
                "      event_to_amplitude_factor_fn:",
                "        name: power_fn_of_velocity",
                "        power: 1.0",
                "      detuning_to_amplitude:",
                "        0.0: 1.0",
                "      random_detuning_range: 0.0",
                "  amplitude_scaling: 1.0",
            ],
            {
                'sine': Instrument(
                    partials=[
                        Partial(
                            wave=ModulatedWave(
                                waveform='sine',
                                amplitude_envelope_fn=create_trapezoid_envelope,
                                phase=0,
                                amplitude_modulator=Modulator(
                                    waveform='sine',
                                    carrier_frequency_ratio=1,
                                    modulator_frequency_ratio=0.99,
                                    modulation_index_envelope_fn=create_trapezoid_envelope,
                                    phase=0,
                                    use_ring_modulation=True
                                ),
                                phase_modulator=None,
                                quasiperiodic_bandwidth=0
                            ),
                            frequency_ratio=1.0,
                            amplitude_ratio=1.0,
                            event_to_amplitude_factor_fn=functools.partial(
                                compute_amplitude_factor_as_power_of_velocity,
                                power=1
                            ),
                            detuning_to_amplitude={0.0: 1.0},
                            random_detuning_range=0.0,
                            effects=[]
                        )
                    ],
                    amplitude_scaling=1.0,
                    effects=[]
                )
            }
        ),
        (
            [
                "---",
                "- name: fm_sine",
                "  partials:",
                "    - wave:",
                "        waveform: sine",
                "        amplitude_envelope_fn:",
                "          name: trapezoid",
                "        phase_modulator:",
                "          waveform: sine",
                "          frequency_ratio_numerator: 3",
                "          frequency_ratio_denominator: 1",
                "          modulation_index_envelope_fn:",
                "            name: generic_ahdsr",
                "            attack_to_ahds_max_ratio: 0.1",
                "            max_attack_duration: 0.05",
                "            attack_degree: 2.0",
                "            hold_to_hds_max_ratio: 0",
                "            max_hold_duration: 0",
                "            decay_to_ds_max_ratio: 1.0",
                "            max_decay_duration: 100",
                "            decay_degree: 0.5",
                "            sustain_level: 0.05",
                "            max_sustain_duration: 0",
                "            max_release_duration: 0.05",
                "            release_duration_on_velocity_order: 0",
                "            release_degree: 1.5",
                "            peak_value: 5.0",
                "            ratio_at_zero_velocity: 0.5",
                "            envelope_values_on_velocity_order: 0.5",
                "      frequency_ratio: 1.0",
                "      amplitude_ratio: 1.0",
                "      event_to_amplitude_factor_fn:",
                "        name: power_fn_of_velocity",
                "        power: 1.0",
                "      detuning_to_amplitude:",
                "        -0.01: 0.5",
                "        0.01: 0.5",
                "      random_detuning_range: 0.0",
                "  amplitude_scaling: 1.0",
                "  effects:",
                "    - name: filter",
                "      kind: absolute",
                "      max_frequency: 10000",
                "      order: 2"
            ],
            {
                'fm_sine': Instrument(
                    partials=[
                        Partial(
                            wave=ModulatedWave(
                                waveform='sine',
                                amplitude_envelope_fn=create_trapezoid_envelope,
                                phase=0,
                                amplitude_modulator=None,
                                phase_modulator=Modulator(
                                    waveform='sine',
                                    carrier_frequency_ratio=1,
                                    modulator_frequency_ratio=3,
                                    modulation_index_envelope_fn=functools.partial(
                                        create_generic_ahdsr_envelope,
                                        attack_to_ahds_max_ratio=0.1,
                                        max_attack_duration=0.05,
                                        attack_degree=2.0,
                                        hold_to_hds_max_ratio=0,
                                        max_hold_duration=0,
                                        decay_to_ds_max_ratio=1.0,
                                        max_decay_duration=100,
                                        decay_degree=0.5,
                                        sustain_level=0.05,
                                        max_sustain_duration=0,
                                        max_release_duration=0.05,
                                        release_duration_on_velocity_order=0,
                                        release_degree=1.5,
                                        peak_value=5.0,
                                        ratio_at_zero_velocity=0.5,
                                        envelope_values_on_velocity_order=0.5
                                    ),
                                    phase=0,
                                    use_ring_modulation=False
                                ),
                                quasiperiodic_bandwidth=0
                            ),
                            frequency_ratio=1.0,
                            amplitude_ratio=1.0,
                            event_to_amplitude_factor_fn=functools.partial(
                                compute_amplitude_factor_as_power_of_velocity,
                                power=1
                            ),
                            detuning_to_amplitude={-0.01: 0.5, 0.01: 0.5},
                            random_detuning_range=0.0,
                            effects=[]
                        )
                    ],
                    amplitude_scaling=1.0,
                    effects=[
                        functools.partial(
                            apply_frequency_filter,
                            kind='absolute', max_frequency=10000, order=2
                        )
                    ]
                )
            }
        ),
    ]
)
def test_create_instruments_registry(
        path_to_tmp_file: str, yaml_content: List[str],
        expected: Dict[str, Instrument]
) -> None:
    """Test `create_instruments_registry` function."""
    with open(path_to_tmp_file, 'w') as tmp_yml_file:
        for line in yaml_content:
            tmp_yml_file.write(line + '\n')
    result = create_instruments_registry(path_to_tmp_file)

    velocities = [0.25, 0.5, 1.0]
    for instrument_name, instrument in expected.items():
        events = [
            Event(
                instrument=instrument_name,
                start_time=0.0,
                duration=1.0,
                frequency=440,
                velocity=velocity,
                effects='',
                frame_rate=8000
            )
            for velocity in velocities
        ]
        for event in events:
            resulting_sound = synthesize(event, result)
            expected_sound = synthesize(event, expected)
            np.testing.assert_allclose(resulting_sound, expected_sound)

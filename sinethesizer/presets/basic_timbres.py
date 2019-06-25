"""
Define basic timbres.

Author: Nikolay Lysenko
"""


from typing import Tuple

from sinethesizer.synth.timbres import TimbreSpec, OvertoneSpec
from sinethesizer.presets.basic_envelopes import constant_with_linear_decrease


sine = TimbreSpec(
    fundamental_waveform='sine',
    fundamental_volume_envelope_fn=constant_with_linear_decrease,
    overtones_specs=[]
)


def define_sine_with_n_harmonics(n: int) -> Tuple[str, TimbreSpec]:
    """
    Define a parametric family of timbres called 'sine_with_n_harmonics'.

    :param n:
        number of harmonics
    :return:
        name of timbre and its specification
    """
    timbre_name = f'sine_with_{n}_harmonics'
    timbre_spec = TimbreSpec(
        fundamental_waveform='sine',
        fundamental_volume_envelope_fn=constant_with_linear_decrease,
        overtones_specs=[
            OvertoneSpec(
                waveform='sine',
                frequency_ratio=i,
                volume_share=0.2 / 2 ** (i-2),
                volume_envelope_fn=constant_with_linear_decrease
            )
            for i in range(2, n+2)
        ]
    )
    return timbre_name, timbre_spec

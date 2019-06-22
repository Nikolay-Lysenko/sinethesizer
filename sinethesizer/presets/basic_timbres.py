"""
Define basic timbres.

Author: Nikolay Lysenko
"""


from sinethesizer.synth.timbres import TimbreSpec, OvertoneSpec
from sinethesizer.presets.basic_envelopes import constant_with_linear_decrease


sine_with_harmonics = TimbreSpec(
    fundamental_waveform='sine',
    fundamental_volume_envelope_fn=constant_with_linear_decrease,
    overtones_specs=[
        OvertoneSpec(
            waveform='sine',
            frequency_ratio=i,
            volume_share=0.1,
            volume_envelope_fn=constant_with_linear_decrease
        )
        for i in range(2, 4)
    ]
)

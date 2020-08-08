"""
Validate specifications of timbres.

Author: Nikolay Lysenko
"""


from sinethesizer.synth.timbre import TimbreSpec
from sinethesizer.utils.waves import NAME_TO_WAVEFORM


def validate_timbre_spec(timbre_spec: TimbreSpec) -> None:
    """
    Validate specification of a timbre.

    :param timbre_spec:
        specification of a timbre
    :return:
        None
    """
    fundamental_waveform = timbre_spec.fundamental_waveform
    if fundamental_waveform not in NAME_TO_WAVEFORM.keys():
        raise ValueError(
            f"Unknown name of fundamental waveform: {fundamental_waveform}."
        )

    if len(timbre_spec.overtones_specs) == 0:
        return

    for overtone_spec in timbre_spec.overtones_specs:
        if overtone_spec.waveform not in NAME_TO_WAVEFORM.keys():
            raise ValueError(
                f"Unknown name of overtone waveform: {overtone_spec.waveform}."
            )

    min_ratio = min(x.frequency_ratio for x in timbre_spec.overtones_specs)
    if min_ratio <= 1:
        raise ValueError(
            "All overtones must have higher frequencies than the fundamental."
        )

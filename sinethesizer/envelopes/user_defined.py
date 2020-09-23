"""
Create envelopes from user-defined sequences of points.

Author: Nikolay Lysenko
"""


from math import ceil, floor
from typing import Any, Dict, List

import numpy as np


def create_user_defined_envelope(
        event: 'sinethesizer.synth.core.Event', parts: List[Dict[str, Any]],
        ratio_at_zero_velocity: float = 0,
        envelope_values_on_velocity_order: float = 0
) -> np.ndarray:
    """
    Create envelope that is an upsampled version of a user-defined envelope.

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
        and velocity
    :param parts:
        list of dictionaries representing successive parts of an envelope;
        there, 'values' key relates to envelope values at maximum velocity
        (from the very beginning of the envelope up to its very end
        inclusively) and 'max_duration' key relates to maximum allowed duration
        of this part in seconds
    :param ratio_at_zero_velocity:
        ratio of envelope values at zero velocity to envelope values at maximum
        velocity
    :param envelope_values_on_velocity_order:
        coefficient that determines dependence of envelope values on velocity;
        given non-maximum positive velocity, the higher it is, the lower
        envelope values are; if it is 0, velocity does not affect envelope
    :return:
        envelope
    """
    frame_rate = event.frame_rate
    remaining_duration_in_frames = ceil(event.duration * frame_rate)
    # 1 is subtracted, because there are N - 1 intervals between N points.
    remaining_length = sum(len(part['values']) - 1 for part in parts)
    results = []
    for part in parts:
        current_length = len(part['values']) - 1
        length_fraction = current_length / remaining_length
        part_duration_in_frames = min(
            floor(length_fraction * remaining_duration_in_frames),
            floor((part['max_duration'] or 1e7) * frame_rate)
        )

        upsampling_ratio = (part_duration_in_frames - 1) / current_length
        indices = [
            int(round(i * upsampling_ratio)) for i in range(current_length + 1)
        ]
        all_indices = np.linspace(
            0, part_duration_in_frames, part_duration_in_frames, dtype=np.int32
        )
        current_result = np.interp(all_indices, indices, part['values'])
        results.append(current_result)

        remaining_length -= current_length
        remaining_duration_in_frames -= len(current_result)

    envelope = np.concatenate(results)
    coef = event.velocity ** envelope_values_on_velocity_order
    envelope *= ratio_at_zero_velocity + coef * (1 - ratio_at_zero_velocity)
    return envelope

"""
Test `sinethesizer.effects.reverb` module.

Author: Nikolay Lysenko
"""


import math
from typing import Optional, Sequence

import numpy as np
import pytest

from sinethesizer.effects.reverb import (
    Listener,
    Room,
    SoundSource,
    apply_artificial_reverb,
    apply_room_reverb,
    generate_room_impulse_response,
    generate_tiling,
)
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, event, first_reflection_delay, decay_duration, "
    "amplitude_random_range, n_early_reflections, early_reflections_delay, "
    "diffusion_delay_factor, diffusion_delay_random_range, "
    "late_reflections_decay_power, original_sound_gain, reverberations_gain, "
    "random_seeds, keep_peak_volume, expected",
    [
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 5, 10, 5, 1],
                [1, 5, 10, 5, 4, 3, 2, 1]
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0,
                duration=1,
                frequency=440,
                velocity=1,
                effects='',
                frame_rate=4
            ),
            # `first_reflection_delay`
            2.0,
            # `decay_duration`
            1.8,
            # `amplitude_random_range`
            0.0,
            # `n_early_reflections`
            2,
            # `early_reflections_delay`
            1.0,
            # `diffusion_delay_factor`
            0.5,
            # `diffusion_delay_random_range`
            0.0,
            # `late_reflections_decay_power`
            10.0,
            # `original_sound_gain`
            1.0,
            # `reverberations_gain`
            1.0,
            # `random_seeds`
            (0, 0),
            # `keep_peak_volume`
            False,
            # `expected`
            np.array([
                [
                    1, 2, 3, 4, 5, 10, 5, 1,
                    0.00517892437, 0.0103578487, 0.0155367731, 0.0207156975,
                    0.0262673218, 0.0525346436, 0.0271127032, 0.00692147148,
                    0.00226701291, 0.00428227858, 0.00257054458, 0.00163143618,
                    0.00101775133, 0.000358903552, 0.000051784429
                ],
                [
                    1, 5, 10, 5, 4, 3, 2, 1,
                    0.00517892437, 0.0258946219, 0.0517892437, 0.0258946219,
                    0.0210883974, 0.0174002729, 0.0141848298, 0.00759411567,
                    0.00274953608, 0.00213585123, 0.00140424771, 0.0008797819,
                    0.000355316102, 0.000203550265, 0.000051784429
                ]
            ])
        ),
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 5, 10, 5, 1],
                [1, 5, 10, 5, 4, 3, 2, 1]
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0,
                duration=1,
                frequency=440,
                velocity=1,
                effects='',
                frame_rate=4
            ),
            # `first_reflection_delay`
            2.0,
            # `decay_duration`
            1.8,
            # `amplitude_random_range`
            0.0,
            # `n_early_reflections`
            2,
            # `early_reflections_delay`
            1.0,
            # `diffusion_delay_factor`
            0.5,
            # `diffusion_delay_random_range`
            0.0,
            # `late_reflections_decay_power`
            10.0,
            # `original_sound_gain`
            1.0,
            # `reverberations_gain`
            1.0,
            # `random_seeds`
            (0, 0),
            # `keep_peak_volume`
            True,
            # `expected`
            np.array([
                [
                    1, 2, 3, 4, 5, 10, 5, 1,
                    0.00517892437, 0.0103578487, 0.0155367731, 0.0207156975,
                    0.0262673218, 0.0525346436, 0.0271127032, 0.00692147148,
                    0.00226701291, 0.00428227858, 0.00257054458, 0.00163143618,
                    0.00101775133, 0.000358903552, 0.000051784429
                ],
                [
                    1, 5, 10, 5, 4, 3, 2, 1,
                    0.00517892437, 0.0258946219, 0.0517892437, 0.0258946219,
                    0.0210883974, 0.0174002729, 0.0141848298, 0.00759411567,
                    0.00274953608, 0.00213585123, 0.00140424771, 0.0008797819,
                    0.000355316102, 0.000203550265, 0.000051784429
                ]
            ])
        ),
    ]
)
def test_apply_artificial_reverb(
        sound: np.ndarray, event: Event,
        first_reflection_delay: float,
        decay_duration: float,
        amplitude_random_range: float,
        n_early_reflections: int,
        early_reflections_delay: float,
        diffusion_delay_factor: float,
        diffusion_delay_random_range: float,
        late_reflections_decay_power: float,
        original_sound_gain: float,
        reverberations_gain: float,
        keep_peak_volume: bool,
        random_seeds: Sequence[Optional[int]],
        expected: np.ndarray
) -> None:
    """Test `apply_artificial_reverb` function."""
    result = apply_artificial_reverb(
        sound, event, first_reflection_delay, decay_duration,
        amplitude_random_range, n_early_reflections, early_reflections_delay,
        diffusion_delay_factor, diffusion_delay_random_range,
        late_reflections_decay_power, original_sound_gain, reverberations_gain,
        random_seeds, keep_peak_volume
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "sound, event, room_length, room_width, room_height, reflection_decay_factor, sound_speed, "
    "listener_x, listener_y, listener_z, listener_direction_x, listener_direction_y, "
    "sound_source_x, sound_source_y, sound_source_z, "
    "sound_source_direction_x, sound_source_direction_y, sound_source_direction_z, "
    "angle, n_reflections, expected",
    [
        (
            # `sound`
            np.array([
                [1, 0, 0],
                [1, 0, 0],
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0,
                duration=1,
                frequency=440,
                velocity=1,
                effects='',
                frame_rate=20
            ),
            # `room_length`
            4,
            # `room_width`
            5,
            # `room_height`
            3,
            # `reflection_decay_factor`
            0.8,
            # `sound_speed`
            10,
            # `listener_x`
            3,
            # `listener_y`
            2,
            # `listener_z`
            1.5,
            # `listener_direction_x`
            0,
            # `listener_direction_y`
            1,
            # `sound_source_x`
            2,
            # `sound_source_y`
            4,
            # `sound_source_z`
            1,
            # `sound_source_direction_x`
            0,
            # `sound_source_direction_y`
            -1,
            # `sound_source_direction_z`
            0,
            # `angle`
            math.pi,
            # `n_reflections`
            1,
            # `expected`
            np.array([
                [0.85065081, 0, 0.28731418, 0.48196489, 0.50363428, 0, 0.06409372, 0, 0.19412925, 0, 0],
                [0.52573111, 0, 0.4648841, 0.14592723, 0.72329581, 0, 0.33281194, 0, 0.2291619, 0, 0],
            ])
        ),
        (
            # `sound`
            np.array([
                [1, 0, 0],
                [0, 0.8, 0],
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0,
                duration=1,
                frequency=440,
                velocity=1,
                effects='',
                frame_rate=20
            ),
            # `room_length`
            4,
            # `room_width`
            5,
            # `room_height`
            3,
            # `reflection_decay_factor`
            0.8,
            # `sound_speed`
            10,
            # `listener_x`
            3,
            # `listener_y`
            2,
            # `listener_z`
            1.5,
            # `listener_direction_x`
            0,
            # `listener_direction_y`
            1,
            # `sound_source_x`
            2,
            # `sound_source_y`
            4,
            # `sound_source_z`
            1,
            # `sound_source_direction_x`
            0,
            # `sound_source_direction_y`
            -1,
            # `sound_source_direction_z`
            0,
            # `angle`
            math.pi,
            # `n_reflections`
            2,
            # `expected`
            np.array([
                [
                    0.4253254, 0.34026032, 0.14365709, 0.35590811, 0.49303024, 0.4675896,
                    0.32081997, 0.3374419, 0.4979458, 0.56634129, 0.26419902, 0.09599053,
                    0.13592096, 0.05443826, 0.07893715, 0.06314972, 0, 0, 0, 0, 0.04476926,
                    0.03581541, 0
                ],
                [
                    0.26286556, 0.21029244, 0.23244205, 0.25891725, 0.5799628, 0.78101759,
                    0.54083325, 0.28814954, 0.24458103, 0.42756045, 0.33022196, 0.11934283,
                    0.09186148, 0.04805713, 0.00866511, 0.00693209, 0, 0, 0, 0, 0.04119367,
                    0.03295493, 0
                ]
            ])
        ),
    ]
)
def test_apply_room_reverb(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        room_length: float, room_width: float, room_height: float,
        reflection_decay_factor: float, sound_speed: float,
        listener_x: float, listener_y: float, listener_z: float,
        listener_direction_x: float, listener_direction_y: float,
        sound_source_x: float, sound_source_y: float, sound_source_z: float,
        sound_source_direction_x: float, sound_source_direction_y: float,
        sound_source_direction_z: float, angle: float,
        n_reflections: int, expected
) -> None:
    """Test `apply_room_reverb` function."""
    result = apply_room_reverb(
        sound, event, room_length, room_width, room_height, reflection_decay_factor, sound_speed,
        listener_x, listener_y, listener_z, listener_direction_x, listener_direction_y,
        sound_source_x, sound_source_y, sound_source_z,
        sound_source_direction_x, sound_source_direction_y, sound_source_direction_z,
        angle, n_reflections
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "room, listener, sound_source, n_reflections, frame_rate, expected",
    [
        (
            # `room`
            Room((4, 5, 3), 0.8),
            # `listener`
            Listener((1, 2, 1.5), (0, 1)),
            # `sound_source`
            SoundSource((1 + 1 / math.sqrt(3), 3, 1.5), (0, -1, 0), math.pi / 2),
            # `n_reflections`
            0,
            # `frame_rate`
            12,
            # `expected`
            np.array([
                [math.sin(math.pi / 6)],
                [math.cos(math.pi / 6)],
            ])
        ),
        (
            # `room`
            Room((4, 5, 3), 0.8),
            # `listener`
            Listener((1, 2, 1.5), (0, 1)),
            # `sound_source`
            SoundSource((2, 3, 1.5), (0, -1, 0), math.pi),
            # `n_reflections`
            0,
            # `frame_rate`
            12,
            # `expected`
            np.array([
                [math.sin(math.pi / 8)],
                [math.cos(math.pi / 8)],
            ])
        ),
        (
            # `room`
            Room((4, 5, 3), 0.8),
            # `listener`
            Listener((2, 3, 1.5), (1, 1)),
            # `sound_source`
            SoundSource((3, 4, 1.5), (1, 1, 0), 0.001),
            # `n_reflections`
            2,
            # `frame_rate`
            12,
            # `expected`
            np.array([
                [0.64 / 3 / math.sqrt(2)],
                [0.64 / 3 / math.sqrt(2)],
            ])
        ),
        (
            # `room`
            Room((4, 5, 3), 0.8, 10),
            # `listener`
            Listener((3, 2, 1.5), (0, 1)),
            # `sound_source`
            SoundSource((2, 4, 1), (0, -1, 0), math.pi),
            # `n_reflections`
            1,
            # `frame_rate`
            20,
            # `expected`
            np.array([
                [0.85065081, 0, 0.28731418, 0.48196489, 0.50363428, 0, 0.06409372, 0, 0.19412925],
                [0.52573111, 0, 0.4648841, 0.14592723, 0.72329581, 0, 0.33281194, 0, 0.2291619],
            ])
        ),
    ]
)
def test_generate_room_impulse_response(
        room: Room, listener: Listener, sound_source: SoundSource,
        n_reflections: int, frame_rate: int, expected: np.ndarray
) -> None:
    """Test `generate_room_impulse_response` function."""
    result = generate_room_impulse_response(
        room, listener, sound_source, n_reflections, frame_rate
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "room, listener, n_reflections, expected",
    [
        (
            # `room`
            Room((4, 5, 3), 0.8),
            # `listener`
            Listener((1, 2, 1.5), (0, 1)),
            # `n_reflections`
            2,
            # expected`
            {
                0: {
                    (0, 0, 0): np.array([1, 2, 1.5]),
                },
                1: {
                    (1, 0, 0): np.array([7, 2, 1.5]),
                    (-1, 0, 0): np.array([-1, 2, 1.5]),
                    (0, 1, 0): np.array([1, 8, 1.5]),
                    (0, -1, 0): np.array([1, -2, 1.5]),
                    (0, 0, 1): np.array([1, 2, 4.5]),
                    (0, 0, -1): np.array([1, 2, -1.5]),
                },
                2: {
                    (2, 0, 0): np.array([9, 2, 1.5]),
                    (-2, 0, 0): np.array([-7, 2, 1.5]),
                    (1, 1, 0): np.array([7, 8, 1.5]),
                    (-1, 1, 0): np.array([-1, 8, 1.5]),
                    (0, 2, 0): np.array([1, 12, 1.5]),
                    (1, -1, 0): np.array([7, -2, 1.5]),
                    (-1, -1, 0): np.array([-1, -2, 1.5]),
                    (0, -2, 0): np.array([1, -8, 1.5]),
                    (1, 0, 1): np.array([7, 2, 4.5]),
                    (-1, 0, 1): np.array([-1, 2, 4.5]),
                    (0, 1, 1): np.array([1, 8, 4.5]),
                    (0, -1, 1): np.array([1, -2, 4.5]),
                    (0, 0, 2): np.array([1, 2, 7.5]),
                    (1, 0, -1): np.array([7, 2, -1.5]),
                    (-1, 0, -1): np.array([-1, 2, -1.5]),
                    (0, 1, -1): np.array([1, 8, -1.5]),
                    (0, -1, -1): np.array([1, -2, -1.5]),
                    (0, 0, -2): np.array([1, 2, -4.5]),
                },
            }
        ),
    ]
)
def test_generate_tiling(
        room: Room, listener: Listener, n_reflections: int,
        expected: dict[int,  dict[tuple[float, ...], np.ndarray]]
) -> None:
    """Test `generate_tiling` function."""
    result = generate_tiling(room, listener, n_reflections)
    assert result.keys() == expected.keys()
    for (_, nested_result), (_, nested_expected) in zip(result.items(), expected.items()):
        assert nested_result.keys() == nested_expected.keys()
        zipped = zip(nested_result.items(), nested_expected.items())
        for (_, result_value), (_, expected_value) in zipped:
            np.testing.assert_equal(result_value, expected_value)

"""
Imitate acoustic reflections within a room.

Author: Nikolay Lysenko
"""


import math
import random
import warnings
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
from scipy.signal import convolve


def validate_inputs(
        decay_duration: float, n_early_reflections: int,
        early_reflections_delay: float, diffusion_delay_factor: float
) -> None:
    """
    Validate arguments passed to `apply_artificial_reverb` function.

    :param decay_duration:
        total time (in seconds) since the first reflection until the last one
    :param n_early_reflections:
        number of equally spaced reflections
    :param early_reflections_delay:
        time (in seconds) between successive early reflections
    :param diffusion_delay_factor:
        exponential decay factor for time between successive late reflections
    :return:
        None
    """
    early_span = (n_early_reflections - 1) * early_reflections_delay
    late_reflections_duration = decay_duration - early_span
    if late_reflections_duration < 0:
        raise ValueError("Early reflections last more than all reflections.")
    if diffusion_delay_factor < 0:
        raise ValueError("Diffusion delay factor can not be negative.")
    max_late_reflections_duration = (
        # Right multiplier is sum of geometric progression: a + a^2 + a^3 + ...
        early_reflections_delay * (1 / (1 - diffusion_delay_factor) - 1)
        if diffusion_delay_factor < 1 else np.inf
    )
    padding = 0.05  # Due to randomness, sum of the series may be less.
    if max_late_reflections_duration < late_reflections_duration + padding:
        raise ValueError(
            "Inconsistent parameters. "
            "Increase either `diffusion_delay_factor` or `early_reflections_delay` "
            "or decrease `decay_duration`."
        )
    if diffusion_delay_factor > 1:  # pragma: no cover
        warnings.warn("If `diffusion_delay_factor` > 1, reverb is not realistic.", UserWarning)


def calculate_reflection_times(
        ir_duration_in_seconds: float,
        first_reflection_delay: float,
        n_early_reflections: int,
        early_reflections_delay: float,
        diffusion_delay_factor: float,
        diffusion_delay_random_range: float,
        random_numbers_generator: random.Random
) -> list[float]:
    """
    Calculate delays with which reflected waves arrive to a listener.

    :param ir_duration_in_seconds:
        duration of impulse response wave (in seconds)
    :param first_reflection_delay:
        time (in seconds) between original sound and its first reflection
    :param n_early_reflections:
        number of equally spaced reflections
    :param early_reflections_delay:
        time (in seconds) between successive early reflections
    :param diffusion_delay_factor:
        exponential decay factor for time between successive late reflections
    :param diffusion_delay_random_range:
        relative range where time between successive late reflections can randomly vary
    :param random_numbers_generator:
        random numbers generator
    :return:
        delays with which reflected waves arrive to a listener
    """
    reflection_times = [first_reflection_delay]
    for i in range(n_early_reflections - 1):
        next_reflection_time = reflection_times[-1] + early_reflections_delay
        reflection_times.append(next_reflection_time)
    delay_without_randomness = early_reflections_delay
    max_n_iterations = int(1e5)
    for _ in range(max_n_iterations):  # pragma: no branch
        delay_without_randomness *= diffusion_delay_factor
        random_number = random_numbers_generator.uniform(
            -diffusion_delay_random_range, diffusion_delay_random_range
        )
        delay = delay_without_randomness * (1 + random_number)
        reflection_time = reflection_times[-1] + delay
        if reflection_time < ir_duration_in_seconds:
            reflection_times.append(reflection_time)
        else:
            break
    return reflection_times


def generate_artificial_impulse_response(
        event: 'sinethesizer.synth.core.Event',
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
        random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate impulse response (IR) for artificial reverb.

    :param event:
        parameters of sound event for which this function is called
    :param first_reflection_delay:
        time (in seconds) between original sound and its first reflection
    :param decay_duration:
        total time (in seconds) since the first reflection until the last one
    :param amplitude_random_range:
        relative range where amplitude of a reflection can randomly vary
    :param n_early_reflections:
        number of equally spaced reflections
    :param early_reflections_delay:
        time (in seconds) between successive early reflections
    :param diffusion_delay_factor:
        exponential decay factor for time between successive late reflections
    :param diffusion_delay_random_range:
        relative range where time between successive late reflections can randomly vary
    :param late_reflections_decay_power:
        power of exponential decay of late reflections amplitude
    :param original_sound_gain:
        fraction of amplitude of original sound that is kept in resulting sound
    :param reverberations_gain:
        fraction of amplitude of reverberations sum that is kept in resulting sound
    :param random_seed:
        seed for pseudo-random number generator
    :return:
        impulse response
    """
    ir_duration_in_seconds = first_reflection_delay + decay_duration
    ir_duration_in_frames = math.ceil(event.frame_rate * ir_duration_in_seconds)
    impulse_response = np.zeros(ir_duration_in_frames)
    impulse_response[0] = original_sound_gain

    random_numbers_generator = random.Random(random_seed)
    reflection_times = calculate_reflection_times(
        ir_duration_in_seconds, first_reflection_delay,
        n_early_reflections, early_reflections_delay,
        diffusion_delay_factor, diffusion_delay_random_range,
        random_numbers_generator
    )
    for reflection_time in reflection_times:
        index = min(
            int(round(event.frame_rate * reflection_time)),
            ir_duration_in_frames - 1
        )
        amplitude = np.e ** -(
            late_reflections_decay_power * reflection_time / ir_duration_in_seconds
        )
        amplitude *= reverberations_gain
        random_factor = 1 + random_numbers_generator.uniform(
            -amplitude_random_range, amplitude_random_range
        )
        amplitude *= random_factor
        impulse_response[index] = amplitude
    return impulse_response


def apply_artificial_reverb(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        first_reflection_delay: float = 0.1,
        decay_duration: float = 1.5,
        amplitude_random_range: float = 0.5,
        n_early_reflections: int = 4,
        early_reflections_delay: float = 0.015,
        diffusion_delay_factor: float = 0.995,
        diffusion_delay_random_range: float = 0.75,
        late_reflections_decay_power: float = 10,
        original_sound_gain: float = 0.8,
        reverberations_gain: float = 0.2,
        random_seeds: Sequence[Optional[int]] = (None, None),
        keep_peak_amplitude: bool = False
) -> np.ndarray:
    """
    Imitate acoustic reflections within an imaginary room.

    This function is suitable for enriching 'thin' sounds rather than for sound spatialization.
    Consider using `apply_room_reverb` function for the latter.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param first_reflection_delay:
        time (in seconds) between original sound and its first reflection
    :param decay_duration:
        total time (in seconds) since the first reflection until the last one
    :param amplitude_random_range:
        relative range where amplitude of a reflection can randomly vary
    :param n_early_reflections:
        number of equally spaced reflections
    :param early_reflections_delay:
        time (in seconds) between successive early reflections
    :param diffusion_delay_factor:
        exponential decay factor for time between successive late reflections
    :param diffusion_delay_random_range:
        relative range where time between successive late reflections can randomly vary
    :param late_reflections_decay_power:
        power of exponential decay of late reflections amplitude
    :param original_sound_gain:
        fraction of amplitude of original sound that is kept  in resulting sound
    :param reverberations_gain:
        fraction of amplitude of reverberations sum that is kept in resulting sound
    :param random_seeds:
        seeds for pseudo-random number generator; one is used for the left channel and another one
        is used for the right channel
    :param keep_peak_amplitude:
        if it is set to `True`, processed sound is rescaled to maintain its original peak amplitude
        which is usually changed due to wave interference
    :return:
        reverberated sound
    """
    validate_inputs(
        decay_duration, n_early_reflections, early_reflections_delay,
        diffusion_delay_factor
    )
    left_ir = generate_artificial_impulse_response(
        event, first_reflection_delay, decay_duration, amplitude_random_range,
        n_early_reflections, early_reflections_delay, diffusion_delay_factor,
        diffusion_delay_random_range, late_reflections_decay_power,
        original_sound_gain, reverberations_gain, random_seeds[0]
    )
    right_ir = generate_artificial_impulse_response(
        event, first_reflection_delay, decay_duration, amplitude_random_range,
        n_early_reflections, early_reflections_delay, diffusion_delay_factor,
        diffusion_delay_random_range, late_reflections_decay_power,
        original_sound_gain, reverberations_gain, random_seeds[1]
    )
    original_peak_amplitude = np.max(np.abs(sound))
    sound = np.vstack((
        convolve(sound[0, :], left_ir),
        convolve(sound[1, :], right_ir)
    ))
    if keep_peak_amplitude:
        sound = original_peak_amplitude / np.max(np.abs(sound)) * sound
    return sound


class Room:
    """
    Room where reverberations happen.

    :param dimensions:
        length (x-axis), width (y-axis), and height (z-axis) of the room (in meters)
    :param reflection_decay_factor:
        ratio of wave amplitude right after a reflection to wave amplitude
        just before the reflection; this parameter controls sound absorption by the room
    :param sound_speed:
        speed of sound in the room (in meters per second)
    """

    def __init__(
            self,
            dimensions: tuple[float, float, float],
            reflection_decay_factor: float,
            sound_speed: float = 343
    ):
        self._dimensions = dimensions
        self._reflection_decay_factor = reflection_decay_factor
        self._sound_speed = sound_speed

    def __eq__(self, other):
        if not isinstance(other, Room):  # pragma: no cover
            return False
        equals = (
            self._dimensions == other._dimensions
            and self._reflection_decay_factor == other._reflection_decay_factor
            and self._sound_speed == other._sound_speed
        )
        return equals

    def __hash__(self):
        return hash((self._dimensions, self._reflection_decay_factor, self._sound_speed))

    @property
    def dimensions(self) -> np.ndarray:
        return np.array(self._dimensions)

    @property
    def reflection_decay_factor(self) -> float:
        return self._reflection_decay_factor

    @property
    def sound_speed(self) -> float:
        return self._sound_speed


class Listener:
    """
    Listener within a room.

    :param location:
        location as a triple of coordinates (in meters)
    :param direction:
        2D vector defining azimuth angle of listener orientation;
        it is implicitly assumed that elevation angle is 0;
        this parameter affects distribution of sound amongst left and right channels
    """

    def __init__(
            self,
            location: tuple[float, float, float],
            direction: tuple[float, float]
    ):
        self._location = location
        self._direction = direction

    def __eq__(self, other):
        if not isinstance(other, Listener):  # pragma: no cover
            return False
        equals = (
            self._location == other._location
            and self._direction == other._direction
        )
        return equals

    def __hash__(self):
        return hash((self._location, self._direction))

    @property
    def location(self) -> np.ndarray:
        return np.array(self._location)

    @property
    def direction(self) -> np.ndarray:
        return np.array(self._direction + (0,))


class SoundSource:
    """
    Sound source that emits sound waves equally in all directions that are within a solid angle.

    :param location:
        location as a triple of coordinates (in meters)
    :param direction:
        vector along which the spherical solid angle is oriented
    :param angle:
        plain angle (in radians) specifying solid angle size
    """

    def __init__(
            self,
            location: tuple[float, float, float],
            direction: tuple[float, float, float],
            angle: float
    ):
        self._location = location
        self._direction = direction
        self._angle = angle

    def __eq__(self, other):
        if not isinstance(other, SoundSource):  # pragma: no cover
            return False
        equals = (
            self._location == other._location
            and self._direction == other._direction
            and self._angle == other._angle
        )
        return equals

    def __hash__(self):
        return hash((self._location, self._direction, self._angle))

    @property
    def location(self) -> np.ndarray:
        return np.array(self._location)

    @property
    def direction(self) -> np.ndarray:
        return np.array(self._direction)

    @property
    def angle(self) -> float:
        return self._angle


def generate_new_level_of_tiling(
        previous_level: dict[tuple[int, ...], np.ndarray],
        increment_pairs: tuple[tuple[float, float], ...]
) -> dict[tuple[float, ...], np.ndarray]:
    """
    Generate rooms reflected one more time than previously generated reflected rooms.

    :param previous_level:
        previously generated reflected rooms
    :param increment_pairs:
        tuple where for each coordinate there is a pair of listener location shifts for reflections
        over orthogonal to the axis walls (or floor and ceiling)
    :return:
        rooms reflected ome more time than previous ones
    """
    new_level = {}
    for moves, location in previous_level.items():
        for i, (move, increment_pair) in enumerate(zip(moves, increment_pairs)):
            if move != 0:
                move_increment = int(move / abs(move))
                new_moves = moves[:i] + (move + move_increment,) + moves[i+1:]
                increment = increment_pair[(move + int(move < 0)) % 2]
                new_location = location.copy()
                new_location[i] += move_increment * increment
                new_level[new_moves] = new_location
                break
            else:
                new_moves = moves[:i] + (1,) + moves[i + 1:]
                increment = increment_pair[0]
                new_location = location.copy()
                new_location[i] += increment
                new_level[new_moves] = new_location

                new_moves = moves[:i] + (-1,) + moves[i + 1:]
                increment = increment_pair[1]
                new_location = location.copy()
                new_location[i] -= increment
                new_level[new_moves] = new_location
    return new_level


def generate_tiling(
        room: Room, listener: Listener, n_reflections: int
) -> dict[int, dict[tuple[float, ...], np.ndarray]]:
    """
    Generate data structure suitable for reflections analysis.

    The idea behind this data structure is that instead of analysing reflected ray it is possible
    to analyse its continuation in a reflected room. This is possible, because angle of incidence
    equals to angle of reflection.

    In the output data structure, each number of reflections is mapped to dictionary where:
    * keys are triples of reflection counts over corresponding to axes walls (or floor or ceiling)
      with signs corresponding to order of opposite walls (or floor and ceiling),
    * values are coordinates of reflected listener location.

    :param room:
        room where reverberations happen
    :param listener:
        listener within a room
    :param n_reflections:
        number of reflections to be analysed
    :return:
        data structure suitable for reflections analysis
    """
    increment_pairs = tuple(
        (2 * (room.dimensions[i] - listener.location[i]), 2 * listener.location[i])
        for i in range(len(room.dimensions))
    )
    previous_level = {tuple(0 for _ in range(len(room.dimensions))): listener.location}
    tiling = {0: previous_level}
    for i in range(1, n_reflections + 1):
        new_level = generate_new_level_of_tiling(previous_level, increment_pairs)
        tiling[i] = new_level
        previous_level = new_level
    return tiling


@lru_cache(maxsize=100)
def generate_room_impulse_response(
        room: Room, listener: Listener, sound_source: SoundSource,
        n_reflections: int, frame_rate: int
) -> np.ndarray:
    """
    Generate room impulse response.

    :param room:
        room where reverberations happen
    :param listener:
        listener within a room
    :param sound_source:
        sound source that emits sound waves equally in all directions that are within a solid angle
    :param n_reflections:
        number of reflections to be analysed
    :param frame_rate:
        number of frames per second
    :return:
        two-channel impulse response
    """
    straight_distance = np.linalg.norm(sound_source.location - listener.location)
    unit_sound_source_direction = sound_source.direction / np.linalg.norm(sound_source.direction)

    impulse_response_data = []
    tiling = generate_tiling(room, listener, n_reflections)
    for level_n_reflections, level in tiling.items():
        reflection_decay_factor = room.reflection_decay_factor ** level_n_reflections
        for move, location in level.items():
            trajectory = location - sound_source.location
            distance = np.linalg.norm(trajectory)
            unit_trajectory = trajectory / distance
            cosine = np.dot(unit_sound_source_direction, unit_trajectory)
            cosine = np.clip(cosine, -1, 1)  # Prevent issues with numeric overflow.
            angle = math.acos(cosine)
            if angle > sound_source.angle:
                continue  # No sound waves are emitted towards `location`.

            delay = (distance - straight_distance) / room.sound_speed

            distance_decay_factor = straight_distance / distance
            decay_factor = reflection_decay_factor * distance_decay_factor

            reflections_orientation = (-np.ones_like(listener.direction)) ** np.array(np.abs(move))
            listener_direction = listener.direction * reflections_orientation
            unit_listener_direction = listener_direction / np.linalg.norm(listener_direction)
            projected_trajectory = np.append(trajectory[:-1], 0)
            unit_projected_trajectory = projected_trajectory / np.linalg.norm(projected_trajectory)
            sine = np.cross(unit_listener_direction, unit_projected_trajectory)[-1]
            sine = np.clip(sine, -1, 1)
            angle = math.asin(sine)
            rescaled_angle = 0.5 * (angle + math.pi / 2)
            left_value = decay_factor * math.cos(rescaled_angle)
            right_value = decay_factor * math.sin(rescaled_angle)
            if move[-1] % 2 == 1:  # If listener is located upwards down, swap channels.
                left_value, right_value = right_value, left_value

            impulse_response_data.append({'delay': delay, 'values': (left_value, right_value)})

    max_delay = max(x['delay'] for x in impulse_response_data)
    ir_duration_in_frames = int(round(frame_rate * max_delay)) + 1
    impulse_response = np.zeros((2, ir_duration_in_frames))
    for record in impulse_response_data:
        index = min(
            int(round(frame_rate * record['delay'])),
            ir_duration_in_frames - 1
        )
        impulse_response[0, index] += record['values'][0]
        impulse_response[1, index] += record['values'][1]
    return impulse_response


def apply_room_reverb(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        room_length: float = 12, room_width: float = 7, room_height: float = 5,
        reflection_decay_factor: float = 0.8, sound_speed: float = 343,
        listener_x: float = 3, listener_y: float = 3.5, listener_z: float = 1.75,
        listener_direction_x: float = 1, listener_direction_y: float = 0,
        sound_source_x: float = 9, sound_source_y: float = 3.5, sound_source_z: float = 1.5,
        sound_source_direction_x: float = 1, sound_source_direction_y: float = 0,
        sound_source_direction_z: float = 0, angle: float = math.pi / 2,
        n_reflections: int = 10
) -> np.ndarray:
    """
    Model acoustic reflections within a room.

    By its own, the reverb provided by this function is a stereo reverb, but it is not a so called
    'true stereo' reverb (i.e., reverb that assumes that each channel has its own sound source).
    However, 'true stereo' reverb may be implemented as follows:
    * apply this function to a sound with left channel muted;
    * apply this function to the sound with right channel muted (of course, sound source location
      and/or direction should be changed);
    * sum the outputs.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param room_length:
        length (x-axis) of the room where reverberations happen (in meters)
    :param room_width:
        width (y-axis) of the room where reverberations happen (in meters)
    :param room_height:
        height (z-axis) of the room where reverberations happen (in meters)
    :param reflection_decay_factor:
        ratio of wave amplitude right after a reflection to wave amplitude
        just before the reflection; this parameter controls sound absorption by the room
    :param sound_speed:
        speed of sound in the room (in meters per second)
    :param listener_x:
        x-coordinate of the listener
    :param listener_y:
        y-coordinate of the listener
    :param listener_z:
        z-coordinate of the listener
    :param listener_direction_x:
        projection on x-axis of vector along which the listener is oriented
        within a plane z=const
    :param listener_direction_y:
        projection on y-axis of vector along which the listener is oriented
        within a plane z=const
    :param sound_source_x:
        x-coordinate of the sound source
    :param sound_source_y:
        y-coordinate of the sound source
    :param sound_source_z:
        z-coordinate of the sound source
    :param sound_source_direction_x:
        projection on x-axis of vector along which solid angle of sound emission is oriented
    :param sound_source_direction_y:
        projection on y-axis of vector along which solid angle of sound emission is oriented
    :param sound_source_direction_z:
        projection on z-axis of vector along which solid angle of sound emission is oriented
    :param angle:
        plain angle (in radians) specifying solid angle size
    :param n_reflections:
        number of reflections to be analysed
    :return:
        reverberated_sound
    """
    room_dimensions = (room_length, room_width, room_height)
    room = Room(room_dimensions, reflection_decay_factor, sound_speed)

    listener_location = (listener_x, listener_y, listener_z)
    listener_direction = (listener_direction_x, listener_direction_y)
    listener = Listener(listener_location, listener_direction)

    sound_source_location = (sound_source_x, sound_source_y, sound_source_z)
    sound_source_direction = (
        sound_source_direction_x, sound_source_direction_y, sound_source_direction_z
    )
    sound_source = SoundSource(sound_source_location, sound_source_direction, angle)

    impulse_response = generate_room_impulse_response(
        room, listener, sound_source, n_reflections, event.frame_rate
    )

    if (sound[0, :] == sound[1, :]).all():
        sound = np.vstack((
            convolve(sound[0, :], impulse_response[0, :]),
            convolve(sound[1, :], impulse_response[1, :])
        ))
    else:
        sound = 0.5 * (
            np.vstack((
                convolve(sound[0, :], impulse_response[0, :]),
                convolve(sound[0, :], impulse_response[1, :])
            ))
            + np.vstack((
                convolve(sound[1, :], impulse_response[0, :]),
                convolve(sound[1, :], impulse_response[1, :])
            ))
        )
    return sound

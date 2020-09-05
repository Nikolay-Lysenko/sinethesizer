"""
Synthesize sound.

Author: Nikolay Lysenko
"""


from typing import Any, Callable, Dict, List, NamedTuple

import numpy as np

from sinethesizer.effects import EFFECT_FN_TYPE
from sinethesizer.utils.waves import NAME_TO_WAVEFORM


class ModulatedWave(NamedTuple):
    """
    Specifications of a wave with modulated frequency.

    :param amplitude_envelope_fn:
        function that takes duration (in seconds), velocity, and frame rate
        as inputs and returns amplitude envelope of a resulting wave
    :param carrier_waveform:
        form of a modulated wave (so called carrier)
    :param carrier_phase:
        phase shift of a carrier as a fraction of its period;
        a float between 0 and 1
    :param modulation_index_envelope_fn:
        function that takes duration (in seconds), velocity, and frame rate
        as inputs and returns amplitude envelope of a modulating wave
        (this envelope is known as modulation index envelope)
    :param modulator_waveform:
        form of a modulating wave (so called modulator)
    :param modulator_frequency_ratio:
        ratio of modulator frequency to that of carrier
    :param modulator_phase:
        phase shift of a modulator as a fraction of its period;
        a float between 0 and 1
    """
    amplitude_envelope_fn: Callable[[float, float, int], np.ndarray]
    carrier_waveform: str
    carrier_phase: float
    modulation_index_envelope_fn: Callable[[float, float, int], np.ndarray]
    modulator_waveform: str
    modulator_frequency_ratio: float
    modulator_phase: float


def generate_modulated_wave(
        wave: ModulatedWave, carrier_frequency: float,
        duration_in_seconds: float, velocity: float, frame_rate: int
) -> np.ndarray:
    """
    Generate wave with modulated frequency.

    :param wave:
        specifications of a wave to be generated
    :param carrier_frequency:
        frequency of a carrier wave (in Hz); loosely speaking,
        it is a base frequency of a wave to be generated
    :param duration_in_seconds:
        duration of a wave (in seconds) not including its release;
        in terms of MIDI, it is time between 'NOTE ON' and 'NOTE OFF' events
    :param velocity:
        force of sound generation; it can be likened to force of piano key
        pressing; it is a float between 0 and 1
    :param frame_rate:
        number of frames per second
    :return:
        wave with modulated frequency
    """
    amplitude_envelope = wave.amplitude_envelope_fn(
        duration_in_seconds, velocity, frame_rate
    )
    duration_in_frames = len(amplitude_envelope)

    mod_frequency = wave.modulator_frequency_ratio * carrier_frequency
    mod_period_in_frames = frame_rate / mod_frequency
    mod_phase_in_frames = mod_period_in_frames * wave.modulator_phase
    mod_phase_in_frames = int(round(mod_phase_in_frames))
    mod_xs = np.arange(duration_in_frames) + mod_phase_in_frames

    mod_wave_fn = NAME_TO_WAVEFORM[wave.modulator_waveform]
    modulator = mod_wave_fn(2 * np.pi * mod_frequency / frame_rate * mod_xs)
    modulation_index_envelope = wave.modulation_index_envelope_fn(
        duration_in_seconds, velocity, frame_rate
    )
    modulator *= modulation_index_envelope

    carr_period_in_frames = frame_rate / carrier_frequency
    carr_phase_in_frames = carr_period_in_frames * wave.carrier_phase
    carr_phase_in_frames = int(round(carr_phase_in_frames))
    carr_xs = np.arange(duration_in_frames) + carr_phase_in_frames
    xs = carr_xs + modulator

    carr_wave_fn = NAME_TO_WAVEFORM[wave.carrier_waveform]
    result = carr_wave_fn(2 * np.pi * carrier_frequency / frame_rate * xs)
    result *= amplitude_envelope

    result = np.vstack((result, result))  # Two channels for stereo sound.
    return result


class Partial(NamedTuple):
    """
    Specifications of a partial (fundamental or overtone).

    :param wave:
        specifications of a wave that forms the partial
    :param detune_to_volume:
        mapping from a detuning ratio to relative volume of the corresponding
        detuned frequency; sum of slightly detuned waves sounds less artificial
        than one pure wave
    :param effects:
        sound effects that are always applied to this partial
    """
    wave: ModulatedWave
    detune_to_volume: Dict[float, float]
    effects: List[EFFECT_FN_TYPE]


def sum_two_sounds(
        first_sound: np.ndarray, second_sound: np.ndarray
) -> np.ndarray:
    """
    Sum two sound of probably unequal durations.

    :param first_sound:
        first sound as array of shape (n_channels, n_frames)
    :param second_sound:
        second sound as array of shape (n_channels, n_frames)
    :return:
        sum of the sounds
    """
    first_n_frames = first_sound.shape[1]
    second_n_frames = second_sound.shape[1]
    n_extra_frames = abs(first_n_frames - second_n_frames)
    padding = np.zeros((first_sound.shape[0], n_extra_frames))
    if first_n_frames > second_n_frames:
        second_sound = np.hstack((second_sound, padding))
    elif first_n_frames < second_n_frames:
        first_sound = np.hstack((first_sound, padding))
    return first_sound + second_sound


def generate_partial(
        partial: Partial, frequency: float, duration: float,
        velocity: float, frame_rate: int, context: Dict[str, Any]
) -> np.ndarray:
    """
    Generate partial (fundamental or overtone).

    :param partial:
        specifications of a partial
    :param frequency:
        frequency of the partial (in Hz)
    :param duration:
        duration of a partial (in seconds) not including its release;
        in terms of MIDI, it is time between 'NOTE ON' and 'NOTE OFF' events
    :param velocity:
        force of sound generation; it can be likened to force of piano key
        pressing; it is a float between 0 and 1
    :param frame_rate:
        number of frames per second
    :param context:
        information to be used by sound effects; it can contain number of
        frames per second and fundamental frequency in Hz (if it exists)
    :return:
        partial
    """
    sound = np.array([[], []], dtype=np.float64)
    for frequency_ratio, volume_ratio in partial.detune_to_volume.items():
        detuned_frequency = frequency_ratio * frequency
        wave = generate_modulated_wave(
            partial.wave, detuned_frequency, duration, velocity, frame_rate
        )
        wave *= volume_ratio
        sound = sum_two_sounds(sound, wave)
    for effect_fn in partial.effects:
        sound = effect_fn(sound, context)
    return sound


class Instrument(NamedTuple):
    """
    Specifications of a virtual musical instrument.

    :param partials:
        specifications of partials
    :param frequency_ratios:
        ratios of partial frequencies to the fundamental frequency
    :param partials_volume_fn:
        function that takes position of a partial and velocity as inputs
        and returns ratio of the partial's volume to volume of the fundamental
    :param resonance_filter_fn:
        function that takes sound of resulting series of partials and somehow
        mutes frequencies that are not resonance frequencies of the instrument
    :param effects:
        sound effects that are always applied to outputs of the instrument
    """
    partials: List[Partial]
    frequency_ratios: List[float]
    partials_volume_fn: Callable[[int, float], float]
    resonance_filter_fn: Callable[[np.ndarray], np.ndarray]
    effects: List[EFFECT_FN_TYPE]


def synthesize(
        instrument: Instrument, frequency: float, duration: float,
        volume: float, velocity: float, frame_rate: int
) -> np.ndarray:
    """
    Synthesize one sound event.

    :param instrument:
        specifications of a virtual musical instrument
    :param frequency:
        fundamental frequency of a sound to be synthesized
    :param duration:
        duration of an event (in seconds) not including its release;
        in terms of MIDI, it is time between 'NOTE ON' and 'NOTE OFF' events
    :param volume:
        ratio of the highest amplitude of the resulting sound to maximum
        amplitude that is not clipped by playing devices;
        a float between 0 and 1
    :param velocity:
        force of sound generation; it can be likened to force of piano key
        pressing; it is a float between 0 and 1
    :param frame_rate:
        number of frames per second
    :return:
        synthesized sound
    """
    sound = np.array([[], []], dtype=np.float64)
    context = {
        'frame_rate': frame_rate,
        'fundamental_frequency': frequency
    }
    zipped = zip(instrument.partials, instrument.frequency_ratios)
    for partial_id, (partial, frequency_ratio) in enumerate(zipped):
        partial_frequency = frequency_ratio * frequency
        partial_sound = generate_partial(
            partial, partial_frequency, duration, velocity, frame_rate, context
        )
        partial_sound *= instrument.partials_volume_fn(partial_id, velocity)
        sound = sum_two_sounds(sound, partial_sound)

    sound = instrument.resonance_filter_fn(sound)
    for effect_fn in instrument.effects:
        sound = effect_fn(sound, context)

    sound *= volume / np.max(np.abs(sound))
    return sound

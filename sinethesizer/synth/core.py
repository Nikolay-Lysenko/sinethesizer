"""
Synthesize sound.

Author: Nikolay Lysenko
"""


import json
import random
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from sinethesizer.effects import EFFECT_FN_TYPE, get_effects_registry
from sinethesizer.envelopes import ENVELOPE_FN_TYPE
from sinethesizer.synth.event_to_amplitude_factor import (
    EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE
)
from sinethesizer.oscillators import generate_mono_wave


class Event(NamedTuple):
    """
    Parameters of a basic audio event (loosely speaking, a played note).

    :param instrument:
        name of instrument that is used for synthesis
    :param start_time:
        start time (in seconds) of sound from beginning of its track
    :param duration:
        duration of a sound (in seconds) not including its release;
        in terms of MIDI, it is time between 'NOTE ON' and 'NOTE OFF' events
    :param frequency:
        fundamental frequency of a sound to be synthesized
    :param velocity:
        force of sound generation; it can be likened to force of piano key
        pressing; it is a float between 0 and 1; it can affect volume and
        frequency spectrum
    :param effects:
        JSON string representing list of effects that should be applied to
        resulting sound
    :param frame_rate:
        number of frames per second
    """
    instrument: str
    start_time: float
    duration: float
    frequency: float
    velocity: float
    effects: str
    frame_rate: int


class Modulator(NamedTuple):
    """
    Parameters of a wave that modulates phase or amplitude of another wave.

    :param waveform:
        form of a modulating wave
    :param carrier_frequency_ratio:
        ratio of carrier frequency to fundamental frequency of output wave
        produced with this modulator and, maybe, with other modulators of the
        same wave
    :param modulator_frequency_ratio:
        ratio of modulator frequency to fundamental frequency of output wave
        produced with this modulator and, maybe, with other modulators of the
        same wave
    :param modulation_index_envelope_fn:
        function that takes parameters such as duration, velocity, and
        frame rate as inputs and returns amplitude envelope of a modulating
        wave (this envelope is known as modulation index envelope)
    :param phase:
        phase shift of a modulating wave (in radians)
    :param use_ring_modulation:
        if it is set to `True` and amplitude is modulated, ring modulation
        is applied instead of classical amplitude modulation
    """
    waveform: str
    carrier_frequency_ratio: float
    modulator_frequency_ratio: float
    modulation_index_envelope_fn: ENVELOPE_FN_TYPE
    phase: float
    use_ring_modulation: bool


class ModulatedWave(NamedTuple):
    """
    Parameters of a wave with various types of modulation.

    :param waveform:
        form of a modulated wave (so called carrier)
    :param amplitude_envelope_fn:
        function that takes parameters such as duration, velocity, and
        frame rate as inputs and returns amplitude envelope of output wave
        (before amplitude modulation)
    :param phase:
        phase shift of a carrier (in radians)
    :param amplitude_modulator:
        parameters of an amplitude modulating wave
    :param phase_modulator:
        parameters of a phase modulating wave
    :param quasiperiodic_bandwidth:
        bandwidth (in semitones) of instantaneous frequency random changes;
        these changes make output wave quasi-periodic and, hence, more natural
    """
    waveform: str
    amplitude_envelope_fn: ENVELOPE_FN_TYPE
    phase: float
    amplitude_modulator: Optional[Modulator]
    phase_modulator: Optional[Modulator]
    quasiperiodic_bandwidth: float


def adjust_envelope_duration(
        envelope: np.ndarray, required_len: int
) -> np.ndarray:
    """
    Set duration of envelope to a required value.

    :param envelope:
        envelope to be trimmed or extended
    :param required_len:
        required duration of envelope (in frames)
    :return:
        clipped envelope or envelope with propagated last value
    """
    n_absent_frames = required_len - len(envelope)
    if n_absent_frames <= 0:
        return envelope[:required_len]
    padding = envelope[-1] * np.ones(n_absent_frames)
    envelope = np.hstack((envelope, padding))
    return envelope


def introduce_quasiperiodicity(
        phase_modulator: Optional[np.ndarray], n_frames: int, frame_rate: int,
        frequency: float, quasiperiodic_bandwidth: float
) -> np.ndarray:
    """
    Add non-periodic component (here, smoothed noise) to phase modulator.

    :param phase_modulator:
        phase modulator (if it exists)
    :param n_frames:
        required duration of phase modulator
    :param frame_rate:
        number of frames per second
    :param frequency:
        perceived (in other words, central) frequency of output wave
    :param quasiperiodic_bandwidth:
        bandwidth (in semitones) of instantaneous frequency random changes
    :return:
        phase modulator that makes output wave quasi-periodic
    """
    if quasiperiodic_bandwidth == 0:
        return phase_modulator
    semitone = 2 ** (1 / 12)
    half_of_bandwidth = 0.5 * quasiperiodic_bandwidth
    max_deviation_in_hz = frequency * (semitone ** half_of_bandwidth - 1)
    # Instantaneous frequency for PM is carrier frequency plus derivative of
    # phase modulator (i.e., d(phase_modulator)/dt) divided by 2 * pi.
    # If phase modulator is moving average of standard Gaussian noise,
    # its derivative is a Gaussian random variable (with 0 mean and 2 ** 0.5
    # standard deviation) divided by window size and multiplied by frame rate
    # (loosely speaking, dt = 1 / frame_rate).
    # Below `window_size` is set so that three standard deviations of
    # (1 / (2 * pi)) * d(phase_modulator)/dt random variable are equal to
    # `max_deviation_in_hz`.
    n_sigmas = 3
    std_of_sum = 2 ** 0.5
    window_size = int(round(
        n_sigmas * std_of_sum * frame_rate
        / (2 * np.pi * max_deviation_in_hz)
    ))

    # Output of convolution with valid mode of two arrays of size N and M is
    # max(M, N) - min(M, N) + 1. This output must be equal to `n_frames` and
    # M = `window_size`. Let us solve it for N.
    noise_len = n_frames + window_size - 1
    noise = np.random.normal(0, 1, noise_len)
    weights = np.ones(window_size) / window_size
    smoothed_noise = np.convolve(noise, weights, mode='valid')

    if phase_modulator is None:
        return smoothed_noise
    return phase_modulator + smoothed_noise


def generate_modulated_wave(
        wave: ModulatedWave, frequency: float, event: Event
) -> np.ndarray:
    """
    Generate wave with modulated frequency.

    :param wave:
        parameters of a wave to be generated
    :param frequency:
        fundamental frequency of a wave to be generated (in Hz)
    :param event:
        parameters of sound event for which this function is called
    :return:
        wave with modulated frequency
    """
    amplitude_envelope = wave.amplitude_envelope_fn(event)
    n_frames = len(amplitude_envelope)

    carrier_frequency = frequency
    modulators_as_params = {
        'amplitude_modulator': wave.amplitude_modulator,
        'phase_modulator': wave.phase_modulator,
    }
    modulators_as_arrays = {}
    for key, params in modulators_as_params.items():
        modulator_as_array = None
        if params is not None:
            # NB: `carrier_frequency` is overridden below,
            # so order in `modulators_as_params` matters.
            carrier_frequency = params.carrier_frequency_ratio * frequency
            modulator_frequency = params.modulator_frequency_ratio * frequency
            index_envelope = params.modulation_index_envelope_fn(event)
            index_envelope = adjust_envelope_duration(index_envelope, n_frames)
            modulator_as_array = generate_mono_wave(
                params.waveform,
                modulator_frequency,
                index_envelope,
                event.frame_rate,
                params.phase
            )
        modulators_as_arrays[key] = modulator_as_array

    if wave.amplitude_modulator is not None:
        constant = int(not wave.amplitude_modulator.use_ring_modulation)
        modulators_as_arrays['amplitude_modulator'] += constant
    modulators_as_arrays['phase_modulator'] = introduce_quasiperiodicity(
        modulators_as_arrays['phase_modulator'], n_frames, event.frame_rate,
        frequency, wave.quasiperiodic_bandwidth
    )

    result = generate_mono_wave(
        wave.waveform,
        carrier_frequency,
        amplitude_envelope,
        event.frame_rate,
        wave.phase,
        **modulators_as_arrays
    )

    result = np.vstack((result, result))  # Two channels for stereo sound.
    return result


class Partial(NamedTuple):
    """
    Parameters of a partial (fundamental or overtone).

    :param wave:
        parameters of a wave that forms the partial
    :param frequency_ratio:
        ratio of this partial's frequency to fundamental frequency
    :param amplitude_ratio:
        declared ratio of this partial's peak amplitude to peak amplitude
        of the fundamental (both at maximum velocity); actual amplitude ratio
        may be different if effects applied to the partial and to the
        fundamental are not the same
    :param event_to_amplitude_factor_fn:
        function that maps event to its multiplicative contribution to
        partial's amplitude
    :param random_detuning_range:
        range of random detuning (in semitones)
    :param detuning_to_amplitude:
        mapping from additional detuning size (in semitones) to
        amplitude factor of a wave with the corresponding detuned frequency
    :param effects:
        sound effects that should be applied to this partial
    """
    wave: ModulatedWave
    frequency_ratio: float
    amplitude_ratio: float
    event_to_amplitude_factor_fn: EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE
    random_detuning_range: float
    detuning_to_amplitude: Dict[float, float]
    effects: List[EFFECT_FN_TYPE]


def sum_two_sounds(
        first_sound: np.ndarray, second_sound: np.ndarray
) -> np.ndarray:
    """
    Sum two sounds of probably unequal durations.

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


def generate_partial(partial: Partial, event: Event) -> np.ndarray:
    """
    Generate partial (fundamental or overtone).

    :param partial:
        parameters of the partial
    :param event:
        parameters of sound event for which this function is called
    :return:
        partial
    """
    semitone = 2 ** (1 / 12)
    sound = np.array([[], []], dtype=np.float64)
    partial_frequency = partial.frequency_ratio * event.frequency
    nyquist_frequency = event.frame_rate / 2
    if partial_frequency >= nyquist_frequency:
        # This partial can not be heard, but it creates aliasing, so remove it.
        return sound
    borders_of_random_detuning = (
        -partial.random_detuning_range / 2,
        partial.random_detuning_range / 2
    )
    params = partial.detuning_to_amplitude.items()
    for freq_shift_in_semitones, amplitude_ratio in params:
        freq_shift_in_semitones += random.uniform(*borders_of_random_detuning)
        frequency_ratio = semitone ** freq_shift_in_semitones
        detuned_frequency = frequency_ratio * partial_frequency
        wave = generate_modulated_wave(partial.wave, detuned_frequency, event)
        wave *= amplitude_ratio
        sound = sum_two_sounds(sound, wave)
    sound *= partial.amplitude_ratio
    sound *= partial.event_to_amplitude_factor_fn(event)
    for effect_fn in partial.effects:
        sound = effect_fn(sound, event)
    return sound


class Instrument(NamedTuple):
    """
    Parameters of a virtual musical instrument.

    :param partials:
        parameters of partials
    :param amplitude_scaling:
        amplitude factor selected to prevent clipping by playing devices;
        set it to be less than inverse of sum of `amplitude_ratio` parameters
        over all partials (and, if applicable, take into account an increase in
        partials' amplitudes due to effects)
    :param effects:
        sound effects that should be applied to outputs of the instrument
    """
    partials: List[Partial]
    amplitude_scaling: float
    effects: List[EFFECT_FN_TYPE]


def apply_event_level_effects(sound: np.ndarray, event: Event) -> np.ndarray:
    """
    Apply sound effects that are specific to a particular event.

    :param sound:
        sound to be modified
    :param event:
        event for which `sound` has been produced
    :return:
        modified sound
    """
    if not event.effects:
        return sound
    effects_registry = get_effects_registry()
    effects = json.loads(event.effects)
    for effect in effects:
        effect_name = effect.pop('name')
        sound = effects_registry[effect_name](sound, event, **effect)
    return sound


def synthesize(
        event: Event, instruments_registry: Dict[str, Instrument]
) -> np.ndarray:
    """
    Synthesize one sound event (loosely speaking, a played note).

    :param event:
        parameters of sound event to be synthesized
    :param instruments_registry:
        mapping from instrument names to their representations
    :return:
        synthesized sound as pressure deviation timeline
    """
    sound = np.array([[], []], dtype=np.float64)
    instrument = instruments_registry[event.instrument]
    for partial in instrument.partials:
        partial_sound = generate_partial(partial, event)
        sound = sum_two_sounds(sound, partial_sound)
    for effect_fn in instrument.effects:
        sound = effect_fn(sound, event)
    sound *= instrument.amplitude_scaling
    sound = apply_event_level_effects(sound, event)
    return sound

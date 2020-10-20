"""
Generate basic waves with modulations.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import Optional

import numpy as np
import scipy.signal


TWO_PI = 2 * np.pi


def generate_sawtooth_wave(xs: np.ndarray, phase_step: float) -> np.ndarray:
    """
    Generate band-limited sawtooth wave.

    PolyBLEP method is used to remove aliasing. It involves a polynomial
    approximation of BLEP (Band-Limited Heavyside Step function). Values at
    points that are close enough to points of discontinuity, are modified
    based on this approximation. This works, because exactly discontinuity
    brings all high-frequency content to sawtooth wave.

    :param xs:
        angles (in radians) at which sawtooth function is computed
    :param phase_step:
        step of successive phase increments with frequency and frame rate of
        `xs` and regardless any frequency/phase modulations in `xs`
    :return:
        sawtooth wave
    """
    mod_xs = np.mod(xs, TWO_PI)
    poly_blep_residual = np.zeros_like(xs)

    to_the_left_of_discontinuity = mod_xs > TWO_PI - phase_step
    curr_xs = (mod_xs - TWO_PI) / phase_step
    curr_residual = -(curr_xs ** 2 + 2 * curr_xs + 1)
    np.copyto(
        poly_blep_residual,
        curr_residual,
        where=to_the_left_of_discontinuity
    )

    to_the_right_of_discontinuity = mod_xs < phase_step
    curr_xs = mod_xs / phase_step
    curr_residual = (curr_xs ** 2 - 2 * curr_xs + 1)
    np.copyto(
        poly_blep_residual,
        curr_residual,
        where=to_the_right_of_discontinuity
    )

    sawtooth_wave = scipy.signal.sawtooth(xs) + poly_blep_residual
    return sawtooth_wave


def generate_square_wave(xs: np.ndarray, phase_step: float) -> np.ndarray:
    """
    Generate band-limited square wave.

    PolyBLEP method is used to remove aliasing. It involves a polynomial
    approximation of BLEP (Band-Limited Heavyside Step function). Values at
    points that are close enough to points of discontinuity, are modified
    based on this approximation. This works, because exactly discontinuity
    brings all high-frequency content to square wave.

    :param xs:
        angles (in radians) at which square wave function is computed
    :param phase_step:
        step of successive phase increments with frequency and frame rate of
        `xs` and regardless any frequency/phase modulations in `xs`
    :return:
        square wave
    """
    mod_xs = np.mod(xs, TWO_PI)
    poly_blep_residual = np.zeros_like(xs)

    to_the_left_of_discontinuity_at_zero = mod_xs > TWO_PI - phase_step
    curr_xs = (mod_xs - TWO_PI) / phase_step
    curr_residual = curr_xs ** 2 + 2 * curr_xs + 1
    np.copyto(
        poly_blep_residual,
        curr_residual,
        where=to_the_left_of_discontinuity_at_zero
    )

    to_the_left_of_discontinuity_at_pi = (
        (np.pi - phase_step < mod_xs) & (mod_xs < np.pi)
    )
    curr_xs = (mod_xs - np.pi) / phase_step
    curr_residual = -(curr_xs ** 2 + 2 * curr_xs + 1)
    np.copyto(
        poly_blep_residual,
        curr_residual,
        where=to_the_left_of_discontinuity_at_pi
    )

    to_the_right_of_discontinuity_at_zero = mod_xs < phase_step
    curr_xs = mod_xs / phase_step
    curr_residual = -(curr_xs ** 2 - 2 * curr_xs + 1)
    np.copyto(
        poly_blep_residual,
        curr_residual,
        where=to_the_right_of_discontinuity_at_zero
    )

    to_the_right_of_discontinuity_at_pi = (
        (np.pi <= mod_xs) & (mod_xs < np.pi + phase_step)
    )
    curr_xs = (mod_xs - np.pi) / phase_step
    curr_residual = curr_xs ** 2 - 2 * curr_xs + 1
    np.copyto(
        poly_blep_residual,
        curr_residual,
        where=to_the_right_of_discontinuity_at_pi
    )

    square_wave = scipy.signal.square(xs) + poly_blep_residual
    return square_wave


def generate_triangle_wave(xs: np.ndarray, phase_step: float) -> np.ndarray:
    """
    Generate band-limited triangle wave.

    PolyBLAMP method is used to remove aliasing. It involves a polynomial
    approximation of BLAMP (Band-Limited Ramp function where ramp function is
    an integral of Heavyside step function and its band-limited version is
    an integral of BLEP). Values at points that are close enough to points of
    first derivative non-existance, are modified based on this approximation.
    This works, because exactly discontinuous changes in first derivative bring
    all high-frequency content to triangle wave.

    :param xs:
        angles (in radians) at which triangle wave function is computed
    :param phase_step:
        step of successive phase increments with frequency and frame rate of
        `xs` and regardless any frequency/phase modulations in `xs`
    :return:
        triangle wave
    """
    mod_xs = np.mod(xs, TWO_PI)
    poly_blamp_residual = np.zeros_like(xs)

    near_discontinuity_at_zero = (
        (mod_xs > TWO_PI - 2 * phase_step) | (mod_xs < 2 * phase_step)
    )
    curr_xs = np.minimum(TWO_PI - mod_xs, mod_xs) / phase_step
    curr_residual = phase_step / (15 * TWO_PI) * (
        (2 - curr_xs) ** 5 - 4 * np.clip(1 - curr_xs, 0, None) ** 5
    )
    np.copyto(
        poly_blamp_residual,
        curr_residual,
        where=near_discontinuity_at_zero
    )

    near_discontinuity_at_pi = (
        (np.pi - 2 * phase_step < mod_xs) & (mod_xs < np.pi + 2 * phase_step)
    )
    curr_xs = np.abs(mod_xs - np.pi) / phase_step
    curr_residual = phase_step / (15 * TWO_PI) * (
        4 * np.clip(1 - curr_xs, 0, None) ** 5 - (2 - curr_xs) ** 5
    )
    np.copyto(
        poly_blamp_residual,
        curr_residual,
        where=near_discontinuity_at_pi
    )

    triangle_wave = scipy.signal.sawtooth(xs, width=0.5) + poly_blamp_residual
    return triangle_wave


def generate_power_law_noise(
        xs: np.ndarray, frame_rate: int, psd_decay_order: float,
        n_equalizer_points: int = 300
) -> np.ndarray:
    """
    Generate noise with bandwidth intensity decaying as power of frequency.

    :param xs:
        arrays of input data points; only its length is used
    :param frame_rate:
        number of frames per second in `xs`; it is used for computing
        filter size
    :param psd_decay_order:
        power of frequency in intensity denominator
    :param n_equalizer_points:
        number of points to approximate gain at each frequency
    :return:
        noise
    """
    white_noise = np.random.normal(0, 0.3, xs.shape)
    if psd_decay_order == 0:
        return white_noise
    nyquist_frequency = frame_rate / 2
    audibility_threshold_in_hz = 20
    ratio = audibility_threshold_in_hz / nyquist_frequency
    # Gain at Nyquist frequency must be 0 in order to prevent aliasing.
    breakpoint_frequencies = np.linspace(ratio, 1 - 1e-7, n_equalizer_points)
    gains = 1 / breakpoint_frequencies ** psd_decay_order
    breakpoint_frequencies = np.hstack(
        (np.array([0]), breakpoint_frequencies, np.array([1]))
    )
    gains = np.hstack((gains[:1], gains, np.array([0]))) / gains[0]
    # Below constant is chosen empirically for pink and brown noise.
    # An effect named amplitude normalization can be used for finer control.
    gain_factor = 25
    gains *= gain_factor
    fir_size = 2 * int(round(frame_rate / 100)) + 1
    fir = scipy.signal.firwin2(fir_size, breakpoint_frequencies, gains)
    result = scipy.signal.convolve(white_noise, fir, mode='same')
    return result


def generate_mono_wave(
        waveform: str, frequency: float, amplitude_envelope: np.ndarray,
        frame_rate: int, phase: float = 0,
        amplitude_modulator: Optional[np.ndarray] = None,
        phase_modulator: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate wave with exactly one channel.

    :param waveform:
        form of wave; it can be one of 'sine', 'square', 'triangle',
        'sawtooth', 'white_noise', 'pink_noise', and 'brown_noise'
    :param frequency:
        frequency of wave (in Hz)
    :param amplitude_envelope:
        amplitude envelope; it also defines duration of sound
    :param frame_rate:
        number of frames per second
    :param phase:
        phase shift (in radians)
    :param amplitude_modulator:
        modulator for AM (amplitude modulation) or RM (ring modulation)
    :param phase_modulator:
        modulator for PM (phase modulation)
    :return:
        sound wave as array of shape (1, len(amplitude_envelope))
    """
    duration_in_frames = len(amplitude_envelope)
    moments_in_seconds = np.arange(duration_in_frames) / frame_rate

    name_to_waveform = {
        'sine': np.sin,
        'sawtooth': generate_sawtooth_wave,
        'square': generate_square_wave,
        'triangle': generate_triangle_wave,
        'white_noise': lambda xs: np.random.normal(0, 0.3, xs.shape),
        'pink_noise': partial(generate_power_law_noise, psd_decay_order=1),
        'brown_noise': partial(generate_power_law_noise, psd_decay_order=2),
        # Below waves can be used in effects like filter sweep.
        'raw_sawtooth': scipy.signal.sawtooth,
        'raw_square': scipy.signal.square,
        'raw_triangle': partial(scipy.signal.sawtooth, width=0.5)
    }
    wave_fn = name_to_waveform[waveform]

    phase_step = TWO_PI * frequency / frame_rate
    args_dict = {
        'sawtooth': [phase_step],
        'square': [phase_step],
        'triangle': [phase_step],
        'pink_noise': [frame_rate],
        'brown_noise': [frame_rate],
    }
    args = args_dict.get(waveform, [])

    if phase_modulator is None:
        wave = wave_fn(
            TWO_PI * frequency * moments_in_seconds + phase,
            *args
        )
    else:
        wave = wave_fn(
            TWO_PI * frequency * moments_in_seconds + phase + phase_modulator,
            *args
        )
    wave *= amplitude_envelope
    if amplitude_modulator is not None:
        wave *= amplitude_modulator
    return wave

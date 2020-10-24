"""
Generate basic waveforms of a classical analog synthesizer.

Author: Nikolay Lysenko
"""


import numpy as np
import scipy.signal


TWO_PI = 2 * np.pi


def generate_sawtooth_wave(xs: np.ndarray, xs_step: float) -> np.ndarray:
    """
    Generate band-limited sawtooth wave.

    PolyBLEP method is applied to remove aliasing. It involves a polynomial
    approximation of BLEP (Band-Limited Heavyside Step function). Values at
    points that are close enough to points of discontinuity, are modified
    based on this approximation. The method works, because discontinuity is
    the thing that brings high-frequency content to sawtooth wave.

    :param xs:
        angles (in radians) at which to compute sawtooth wave values
    :param xs_step:
        step of regular phase increments with frequency and frame rate of
        `xs` and regardless any frequency/phase modulations in `xs`;
        this value is known as phase step or phase increment
    :return:
        sawtooth wave
    """
    mod_xs = np.mod(xs, TWO_PI)
    poly_blep_residual = np.zeros_like(xs)

    to_the_left_of_discontinuity = mod_xs > TWO_PI - xs_step
    curr_xs = (mod_xs[to_the_left_of_discontinuity] - TWO_PI) / xs_step
    curr_residual = -(curr_xs ** 2 + 2 * curr_xs + 1)
    np.place(poly_blep_residual, to_the_left_of_discontinuity, curr_residual)

    to_the_right_of_discontinuity = mod_xs < xs_step
    curr_xs = mod_xs[to_the_right_of_discontinuity] / xs_step
    curr_residual = curr_xs ** 2 - 2 * curr_xs + 1
    np.place(poly_blep_residual, to_the_right_of_discontinuity, curr_residual)

    sawtooth_wave = scipy.signal.sawtooth(xs) + poly_blep_residual
    return sawtooth_wave


def generate_square_wave(xs: np.ndarray, xs_step: float) -> np.ndarray:
    """
    Generate band-limited square wave.

    PolyBLEP method is applied to remove aliasing. It involves a polynomial
    approximation of BLEP (Band-Limited Heavyside Step function). Values at
    points that are close enough to points of discontinuity, are modified
    based on this approximation. The method works, because discontinuity is
    the thing that brings high-frequency content to square wave.

    :param xs:
        angles (in radians) at which to compute square wave values
    :param xs_step:
        step of regular phase increments with frequency and frame rate of
        `xs` and regardless any frequency/phase modulations in `xs`;
        this value is known as phase step or phase increment
    :return:
        square wave
    """
    mod_xs = np.mod(xs, TWO_PI)
    poly_blep_residual = np.zeros_like(xs)

    to_the_left_of_zero = mod_xs > TWO_PI - xs_step
    curr_xs = (mod_xs[to_the_left_of_zero] - TWO_PI) / xs_step
    curr_residual = curr_xs ** 2 + 2 * curr_xs + 1
    np.place(poly_blep_residual, to_the_left_of_zero, curr_residual)

    to_the_left_of_pi = ((np.pi - xs_step < mod_xs) & (mod_xs < np.pi))
    curr_xs = (mod_xs[to_the_left_of_pi] - np.pi) / xs_step
    curr_residual = -(curr_xs ** 2 + 2 * curr_xs + 1)
    np.place(poly_blep_residual, to_the_left_of_pi, curr_residual)

    to_the_right_of_zero = mod_xs < xs_step
    curr_xs = mod_xs[to_the_right_of_zero] / xs_step
    curr_residual = -(curr_xs ** 2 - 2 * curr_xs + 1)
    np.place(poly_blep_residual, to_the_right_of_zero, curr_residual)

    to_the_right_of_pi = ((np.pi <= mod_xs) & (mod_xs < np.pi + xs_step))
    curr_xs = (mod_xs[to_the_right_of_pi] - np.pi) / xs_step
    curr_residual = curr_xs ** 2 - 2 * curr_xs + 1
    np.place(poly_blep_residual, to_the_right_of_pi, curr_residual)

    square_wave = scipy.signal.square(xs) + poly_blep_residual
    return square_wave


def generate_triangle_wave(xs: np.ndarray, xs_step: float) -> np.ndarray:
    """
    Generate band-limited triangle wave.

    PolyBLAMP method is applied to remove aliasing. It involves a polynomial
    approximation of BLAMP (Band-Limited Ramp function where ramp function is
    an integral of Heavyside step function and its band-limited version is
    an integral of BLEP). Values at points that are close enough to points of
    first derivative non-existance, are modified based on this approximation.
    This works, because discontinuous changes of first derivative are the
    things that bring high-frequency content to triangle wave.

    :param xs:
        angles (in radians) at which to compute triangle wave values
    :param xs_step:
        step of regular phase increments with frequency and frame rate of
        `xs` and regardless any frequency/phase modulations in `xs`;
        this value is known as phase step or phase increment
    :return:
        triangle wave
    """
    mod_xs = np.mod(xs, TWO_PI)
    poly_blamp_residual = np.zeros_like(xs)

    near_zero = ((mod_xs > TWO_PI - 2 * xs_step) | (mod_xs < 2 * xs_step))
    curr_xs = mod_xs[near_zero]
    curr_xs = np.minimum(TWO_PI - curr_xs, curr_xs) / xs_step
    curr_residual = xs_step / (15 * TWO_PI) * (
        (2 - curr_xs) ** 5 - 4 * np.clip(1 - curr_xs, 0, None) ** 5
    )
    np.place(poly_blamp_residual, near_zero, curr_residual)

    near_pi = ((np.pi - 2 * xs_step < mod_xs) & (mod_xs < np.pi + 2 * xs_step))
    curr_xs = np.abs(mod_xs[near_pi] - np.pi) / xs_step
    curr_residual = xs_step / (15 * TWO_PI) * (
        4 * np.clip(1 - curr_xs, 0, None) ** 5 - (2 - curr_xs) ** 5
    )
    np.place(poly_blamp_residual, near_pi, curr_residual)

    triangle_wave = scipy.signal.sawtooth(xs, width=0.5) + poly_blamp_residual
    return triangle_wave

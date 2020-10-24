"""
Generate basic periodic and non-periodic waves.

Author: Nikolay Lysenko
"""


from . import analog, facade, noise
from .facade import generate_mono_wave


__all__ = ['analog', 'facade', 'generate_mono_wave', 'noise']

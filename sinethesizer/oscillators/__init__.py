"""
Generate basic periodic and non-periodic waves.

Author: Nikolay Lysenko
"""


from . import analog, facade, karplus_strong, noise
from .facade import generate_mono_wave


__all__ = ['analog', 'facade', 'generate_mono_wave', 'karplus_strong', 'noise']

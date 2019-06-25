"""
Register supported timbres.

Author: Nikolay Lysenko
"""


from sinethesizer.presets.basic_timbres import (
    sine, define_sine_with_n_harmonics
)


TIMBRES_REGISTRY = {
    'sine': sine
}

max_n = 5
for n in range(1, max_n + 1):
    current_timbre_name, current_timbre_spec = define_sine_with_n_harmonics(n)
    TIMBRES_REGISTRY[current_timbre_name] = current_timbre_spec

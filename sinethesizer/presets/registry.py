"""
Register supported timbres.

Author: Nikolay Lysenko
"""


from sinethesizer.presets.timbres import (
    sine, poor_organ, define_sine_with_n_harmonics
)


TIMBRES_REGISTRY = {
    'sine': sine,
    'poor_organ': poor_organ
}

max_n = 5
for n in range(1, max_n + 1):
    current_timbre_name, current_timbre_spec = define_sine_with_n_harmonics(n)
    TIMBRES_REGISTRY[current_timbre_name] = current_timbre_spec

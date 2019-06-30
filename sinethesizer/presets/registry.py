"""
Register supported timbres and effects.

Author: Nikolay Lysenko
"""


from sinethesizer.presets.effects import overdrive, tremolo
from sinethesizer.presets.timbres import (
    sine, poor_organ, define_sine_with_n_harmonics
)


EFFECTS_REGISTRY = {
    'overdrive': overdrive,
    'tremolo': tremolo
}


TIMBRES_REGISTRY = {
    'poor_organ': poor_organ,
    'sine': sine
}

max_n = 5
for n in range(1, max_n + 1):
    current_timbre_name, current_timbre_spec = define_sine_with_n_harmonics(n)
    TIMBRES_REGISTRY[current_timbre_name] = current_timbre_spec

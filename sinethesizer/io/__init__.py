"""
Read input files and write resulting sound files.

Author: Nikolay Lysenko
"""


from .events_to_wav import convert_events_to_timeline, write_timeline_to_wav
from .load_presets import create_instruments_registry
from .midi_to_events import convert_midi_to_events
from .tsv_to_events import convert_tsv_to_events


__all__ = [
    'convert_events_to_timeline',
    'convert_midi_to_events',
    'convert_tsv_to_events',
    'create_instruments_registry',
    'write_timeline_to_wav',
]

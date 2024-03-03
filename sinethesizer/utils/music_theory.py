"""
Help to work with note names and other notions from music theory.

Author: Nikolay Lysenko
"""


import itertools
from functools import lru_cache


@lru_cache(maxsize=1)
def get_list_of_notes() -> list[str]:
    """
    Get list of all notes in alphanumeric notation.

    :return:
        list of all notes from the range of standard piano keyboard
    """
    sorted_pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    pitch_class_to_precedence = {x: i for i, x in enumerate(sorted_pitch_classes)}
    notes = list(itertools.product(sorted_pitch_classes, range(0, 9)))
    notes = sorted(notes, key=lambda x: (x[1], pitch_class_to_precedence[x[0]]))
    notes = notes[9:-11]
    notes = [f'{x[0]}{x[1]}' for x in notes]
    return notes


@lru_cache(maxsize=1)
def get_note_to_position_mapping() -> dict[str, int]:
    """
    Get mapping from note to its position on standard piano keyboard.

    :return:
        mapping from note to its position
    """
    notes = get_list_of_notes()
    note_to_position = dict([(x[1], x[0]) for x in enumerate(notes)])
    return note_to_position


def standardize_note(note: str) -> str:
    """
    Replace note with equivalent note that has 0 or 1 sharps and no flats.

    :param note:
        any valid note in alphanumeric notation
    :return:
        note in alphanumeric notation without flats and with not more than 1 sharp
    """
    note_to_position = get_note_to_position_mapping()
    pivot_note = note.replace('#', '').replace('b', '')
    pivot_position = note_to_position[pivot_note]
    n_sharps = note.count('#')
    n_flats = note.count('b')
    position = pivot_position + n_sharps - n_flats
    notes = get_list_of_notes()
    note = notes[position]
    return note


def convert_note_to_frequency(note: str) -> float:
    """
    Convert note to its frequency in Hz.

    Supported notes can contain any number of sharps ('#') and flats ('b'),
    but must be in the range of standard piano keyboard.

    :param note:
        note in alphanumeric notation
    :return:
        frequency in Hz
    """
    note = standardize_note(note)
    note_to_position = get_note_to_position_mapping()
    position = note_to_position[note]
    a4_position = note_to_position['A4']
    a4_frequency = 440
    semitone = 2 ** (1 / 12)
    frequency = a4_frequency * semitone ** (position - a4_position)
    frequency = round(frequency, 10)  # This is done only due to unit tests.
    return frequency

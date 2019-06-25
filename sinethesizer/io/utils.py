"""
Do auxiliary tasks.

Author: Nikolay Lysenko
"""


import itertools
from typing import List


def get_list_of_notes() -> List[str]:
    """
    Get list of all notes in letter notation.

    :return:
        list of all notes
    """
    order_of_letters = {
        'C': 1, 'Cs': 2, 'D': 3, 'Ds': 4, 'E': 5, 'F': 6, 'Fs': 7,
        'G': 8, 'Gs': 9, 'A': 10, 'As': 11, 'B': 12
    }
    notes = list(itertools.product(order_of_letters.keys(), range(0, 9)))
    notes = sorted(notes, key=lambda x: (x[1], order_of_letters[x[0]]))
    notes = notes[9:-11]
    notes = [f'{x[0]}{x[1]}' for x in notes]
    return notes


def convert_note_to_frequency(note: str) -> float:
    """
    Convert note to its frequency.

    :param note:
        note in letter notation
    :return:
        frequency in Hz
    """
    notes = get_list_of_notes()
    note_to_position = dict([(x[1], x[0]) for x in enumerate(notes)])
    position = note_to_position[note]
    a4_position = note_to_position['A4']
    a4_frequency = 440
    semitone = 2 ** (1 / 12)
    frequency = a4_frequency * semitone ** (position - a4_position)
    return frequency

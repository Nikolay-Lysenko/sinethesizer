"""
Convert piano roll to TSV file of special schema.

A piano roll is a `numpy` 2D-array with rows corresponding to notes,
columns corresponding to time steps, and cells containing zeros and ones
and indicating whether a note is played.

Since piano roll is less informative than TSV format of `sinethesizer`,
some options are filled with reasonable constant defaults. In particular,
volume and location are constant.

Author: Nikolay Lysenko
"""


from typing import Any, List, Tuple

import numpy as np
from sinethesizer.io.utils import get_list_of_notes


def find_involved_notes(lowest_note: str, n_notes: int) -> List[str]:
    """
    Find all consecutive notes that are used in piano roll.

    :param lowest_note:
        the lowest involved note
    :param n_notes:
        number of involved notes
    :return:
        requested number of consecutive notes starting from the given note
    """
    all_notes = get_list_of_notes()
    position = 0
    for position, note in enumerate(all_notes):  # pragma: no branch
        if note == lowest_note:
            break
    involved_notes = all_notes[position:(position + n_notes)]
    return involved_notes


def add_events_with_note(
        events: List[Tuple[Any, ...]], row: np.ndarray, note: str,
        row_number: int, step_in_seconds: float
) -> List[Tuple[Any, ...]]:
    """
    Add all events with a particular note to the list of events.

    :param events:
        list of events to be extended
    :param row:
        timeline of playing the note
    :param note:
        letter notation for the note
    :param row_number:
        number of `row` on its piano roll
    :param step_in_seconds:
        duration of one time step of piano roll in seconds
    :return:
        extended list of events
    """
    start_time = None
    for j in range(len(row) + 1):
        if j < len(row) and row[j] == 1 and start_time is None:
            start_time = j * step_in_seconds
        if (j == len(row) or row[j] == 0) and start_time is not None:
            duration = j * step_in_seconds - start_time
            events.append((start_time, duration, note, -row_number))
            start_time = None
    return events


def convert_roll_to_events(
        roll: np.ndarray, roll_notes: List[str], timbre: str,
        step_in_seconds: float, volume: float,
        location: int = 0, effects: str = ''
) -> List[str]:
    """
    Convert piano roll to rows of TSV file.

    :param roll:
        piano roll to be converted
    :param roll_notes:
        notes that are corresponding to rows of piano roll
    :param timbre:
        timbre to be used
    :param step_in_seconds:
        duration of one time step of piano roll in seconds
    :param volume:
        relative volume of sound to be played
    :param location:
        position of imaginary sound source
    :param effects:
        sound effects to be applied to the resulting event
    :return:
        list of events
    """
    events = []
    for i in range(roll.shape[0]):
        row = roll[i, :]
        note = roll_notes[::-1][i]
        events = add_events_with_note(events, row, note, i, step_in_seconds)
    events = sorted(events, key=lambda x: (x[0], x[3], x[1]))
    events = [
        f"{timbre}\t{x[0]}\t{x[1]}\t{x[2]}\t{volume}\t{location}\t{effects}"
        for x in events
    ]
    return events


def write_roll_to_tsv_file(
        roll: np.ndarray, file_path: str, lowest_note: str, timbre: str,
        step_in_seconds: float, volume: float,
        location: int = 0, effects: str = ''
) -> None:
    """
    Convert piano roll to TSV file.

    :param roll:
        piano roll to be converted
    :param file_path:
        path to a file where result is going to be saved
    :param lowest_note:
        note that corresponds to the lowest row of piano roll
    :param timbre:
        timbre to be used
    :param step_in_seconds:
        duration of one time step of piano roll in seconds
    :param volume:
        relative volume of sound to be played
    :param location:
        position of imaginary sound source
    :param effects:
        sound effects to be applied to the resulting event
    :return:
        None
    """
    roll_notes = find_involved_notes(lowest_note, roll.shape[0])
    events = convert_roll_to_events(
        roll, roll_notes, timbre, step_in_seconds, volume, location, effects
    )
    columns = [
        'timbre', 'start_time', 'duration', 'frequency', 'volume',
        'location', 'effects'
    ]
    header = '\t'.join(columns)
    results = [header] + events
    with open(file_path, 'w') as out_file:
        for line in results:
            out_file.write(line + '\n')

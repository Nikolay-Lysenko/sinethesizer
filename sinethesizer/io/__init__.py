"""
Read input files and write resulting sound files.

Author: Nikolay Lysenko
"""


from .json_to_numpy import convert_json_to_timeline
from .numpy_to_wav import write_timeline_to_wav


__all__ = ['convert_json_to_timeline', 'write_timeline_to_wav']

"""
Take user-defined options and run requested tasks.

Author: Nikolay Lysenko
"""


import argparse

from sinethesizer.io import convert_json_to_timeline, write_timeline_to_wav
from sinethesizer.presets import TIMBRES_REGISTRY
from sinethesizer.synth.utils import validate_timbre_spec


def parse_cli_args() -> argparse.Namespace:
    """
    Parse arguments passed via Command Line Interface (CLI).

    :return:
        namespace with arguments
    """
    parser = argparse.ArgumentParser(description='Virtual analog synthesizer')
    parser.add_argument(
        '-i', '--input_path', type=str, required=True,
        help='path to input JSON file with definition of a track to be played'
    )
    parser.add_argument(
        '-o', '--output_path', type=str, required=True,
        help='path to output file where result is going to be saved as WAV'
    )
    parser.add_argument(
        '-s', '--safe_mode', dest='safe', action='store_true',
        help='validate `TIMBRES_REGISTRY` before core tasks'
    )
    parser.set_defaults(safe=False)

    cli_args = parser.parse_args()
    return cli_args


def main():
    """Run all necessary code."""
    cli_args = parse_cli_args()
    if cli_args.safe:
        for _, timbre_spec in TIMBRES_REGISTRY.items():
            validate_timbre_spec(timbre_spec)
    timeline, frame_rate = convert_json_to_timeline(cli_args.input_path)
    write_timeline_to_wav(cli_args.output_path, timeline, frame_rate)


if __name__ == '__main__':
    main()

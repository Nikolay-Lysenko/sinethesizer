"""
Parse user-defined options and run requested tasks.

Author: Nikolay Lysenko
"""


import argparse
import os

import yaml

from sinethesizer.io import (
    convert_tsv_to_timeline, create_timbres_registry, write_timeline_to_wav
)
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
        help='path to input TSV file with definition of a track to be played'
    )
    parser.add_argument(
        '-p', '--presets_path', type=str, required=True,
        help='path to YAML file with definitions of timbres to be used'
    )
    parser.add_argument(
        '-o', '--output_path', type=str, required=True,
        help='path to output file where result is going to be saved as WAV'
    )
    parser.add_argument(
        '-c', '--config_path', type=str, default=None,
        help='path to configuration file'
    )
    parser.add_argument(
        '-s', '--safe_mode', dest='safe', action='store_true',
        help='validate parsed timbres before core tasks'
    )
    parser.set_defaults(safe=False)

    cli_args = parser.parse_args()
    if cli_args.config_path is None:
        cli_args.config_path = os.path.join(
            os.path.dirname(__file__), 'default_config.yml'
        )
    if cli_args.presets_path is None:
        cli_args.presets_path = os.path.join(
            os.path.dirname(__file__), '..', 'presets', 'demo.yml'
        )
    return cli_args


def main():
    """Run all necessary code."""
    cli_args = parse_cli_args()
    timbres_registry = create_timbres_registry(cli_args.presets_path)
    if cli_args.safe:
        for _, timbre_spec in timbres_registry.items():
            validate_timbre_spec(timbre_spec)
    with open(cli_args.config_path) as config_file:
        settings = yaml.safe_load(config_file)
    settings['timbres_registry'] = timbres_registry
    timeline = convert_tsv_to_timeline(cli_args.input_path, settings)
    write_timeline_to_wav(
        cli_args.output_path, timeline, settings['frame_rate']
    )


if __name__ == '__main__':
    main()

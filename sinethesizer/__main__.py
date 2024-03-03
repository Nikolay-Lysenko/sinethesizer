"""
Parse user-defined options and run requested tasks.

Author: Nikolay Lysenko
"""


import argparse
import importlib.resources

import yaml

from sinethesizer.io import (
    convert_events_to_timeline,
    convert_midi_to_events,
    convert_tsv_to_events,
    create_instruments_registry,
    write_timeline_to_wav,
)


def parse_cli_args() -> argparse.Namespace:
    """
    Parse arguments passed via Command Line Interface (CLI).

    :return:
        namespace with arguments
    """
    parser = argparse.ArgumentParser(description='Standalone synthesizer')
    parser.add_argument(
        '-i', '--input_path', type=str, required=True,
        help='path to input TSV or MIDI file with a track to be played'
    )
    parser.add_argument(
        '-p', '--presets_path', type=str, required=True,
        help='path to YAML file with definitions of instruments to be used'
    )
    parser.add_argument(
        '-o', '--output_path', type=str, required=True,
        help='path to output WAV file'
    )
    parser.add_argument(
        '-m', '--midi_config_path', type=str, default=None,
        help='path to YAML file that defines how to interpret MIDI file'
    )
    parser.add_argument(
        '-c', '--config_path', type=str, default=None,
        help='path to configuration file with general settings'
    )
    cli_args = parser.parse_args()
    return cli_args


def main():
    """Run all necessary code."""
    cli_args = parse_cli_args()

    default_config_path = importlib.resources.files("sinethesizer") / "default_config.yml"
    config_path = cli_args.config_path or default_config_path
    with open(config_path) as config_file:
        settings = yaml.safe_load(config_file)

    instruments_registry = create_instruments_registry(cli_args.presets_path)
    settings['instruments_registry'] = instruments_registry

    if cli_args.midi_config_path is not None:
        with open(cli_args.midi_config_path) as midi_config_file:
            settings['midi'] = yaml.safe_load(midi_config_file)

    extension = cli_args.input_path.split('.')[-1].lower()
    if extension == 'tsv':
        events = convert_tsv_to_events(cli_args.input_path, settings)
    elif extension in ['midi', 'mid']:
        events = convert_midi_to_events(cli_args.input_path, settings)
    else:
        raise ValueError(
            "Only input files with extensions tsv, midi, and mid are allowed, "
            f"but found: {extension}."
        )

    timeline = convert_events_to_timeline(events, settings)
    write_timeline_to_wav(cli_args.output_path, timeline, settings['frame_rate'])


if __name__ == '__main__':
    main()

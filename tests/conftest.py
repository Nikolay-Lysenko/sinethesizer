"""
Define fixtures.

Author: Nikolay Lysenko
"""


from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest


@pytest.fixture()
def path_to_tmp_dir() -> str:
    """Get path to empty temporary directory."""
    with TemporaryDirectory() as path_to_tmp_dir:
        yield path_to_tmp_dir


@pytest.fixture()
def path_to_tmp_file() -> str:
    """Get path to empty temporary file."""
    with NamedTemporaryFile() as tmp_file:
        yield tmp_file.name


path_to_another_tmp_file = path_to_tmp_file

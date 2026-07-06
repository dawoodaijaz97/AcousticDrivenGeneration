import pytest

from main.input_formats import resolve_input_format


def test_resolve_input_format_compact():
    key, col = resolve_input_format("compact-seven")
    assert key == "compact-seven"
    assert col == "input_text"


def test_resolve_input_format_raw():
    key, col = resolve_input_format("instructions-raw")
    assert key == "instructions-raw"
    assert col == "instructions_raw"


def test_resolve_input_format_unknown():
    with pytest.raises(ValueError, match="Unknown input format"):
        resolve_input_format("nope")

import pytest
import importlib.util
from pathlib import Path

# Dynamically load the MIT_Beer_Game module
module_path = Path(__file__).parent.parent / "Games" / "2_MIT_Beer_Game" / "MIT_Beer_Game.py"
spec = importlib.util.spec_from_file_location("beer_game_module", str(module_path))
beer_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beer_mod)

safe_parse_json = beer_mod.safe_parse_json
parse_json_with_default = beer_mod.parse_json_with_default


def test_safe_parse_json_clean():
    data = '{"a":1,"b":2}'
    assert safe_parse_json(data) == {"a": 1, "b": 2}


def test_safe_parse_json_wrapped():
    data = 'text before {"x":10, "y":20} text after'
    assert safe_parse_json(data) == {"x": 10, "y": 20}


def test_parse_json_with_default_valid(capfd):
    default = {"foo": "bar"}
    data = '{"foo":"baz"}'
    result = parse_json_with_default(data, default, "test_valid")
    assert result == {"foo": "baz"}
    # ensure no error was printed
    captured = capfd.readouterr()
    assert captured.out == ''


def test_parse_json_with_default_invalid(capfd):
    default = {"foo": "bar"}
    data = 'not a JSON'
    result = parse_json_with_default(data, default, "test_invalid")
    assert result == default
    captured = capfd.readouterr()
    assert "Error parsing JSON in test_invalid" in captured.out 
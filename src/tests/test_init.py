import pytest

import tests


def test_get_global_config():
    config = tests.read_test_config()
    # yaml.safe_load() returns a list of items/dicts
    assert type(config) == list, "Expected a list"

    # test loading of a specific section
    inputfile_config = tests.read_test_config(section="inputfile")
    assert type(inputfile_config) in (
        list,
        dict,
    ), "Expected a list/or dict for section inputfile"

    # test raising of ValueError exception for invalid section name
    with pytest.raises(expected_exception=ValueError):
        invalid_config = tests.read_test_config(section="does_not_exist!")

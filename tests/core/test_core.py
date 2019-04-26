import yieldengine.core
import pytest


def test_get_global_config():
    import yieldengine
    config = yieldengine.core.get_global_config()
    # yaml.safe_load() returns a list of items/dicts
    assert type(config) == list, "Expected a list"

    # test loading of a specific section
    inputfile_config = yieldengine.core.get_global_config(section="inputfile")
    assert type(inputfile_config) in (list, dict), "Expected a list/or dict for section inputfile"

    # test raising of ValueError exception for invalid section name
    with pytest.raises(expected_exception=ValueError):
        invalid_config = yieldengine.core.get_global_config(section="does_not_exist!")

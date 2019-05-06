import yaml

from tests.paths import TEST_CONFIG_YML


def read_test_config(section: str = None):
    config_file_path = TEST_CONFIG_YML
    # todo: handle exception
    # todo: validate yaml structure (known/required keys)
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
        if section is None:
            return config
        else:
            for element in config:
                if type(element) == dict:
                    for key in element.keys():
                        if key == section:
                            return element[key]

            raise ValueError("Section %s not found in global config!" % section)

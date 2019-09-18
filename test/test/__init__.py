# noinspection PyPackageRequirements
import yaml

from test.paths import TEST_CONFIG_YML


def read_test_config(section: str = None):
    """
    Read the test configuration from a YAML file
    """
    config_file_path = TEST_CONFIG_YML
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

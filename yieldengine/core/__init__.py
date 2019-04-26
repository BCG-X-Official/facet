import os
import yaml
import yieldengine


def get_global_config(section: str = None):
    base_dir = os.path.dirname(yieldengine.__file__)
    config_file_path = os.path.join(base_dir, "yield_engine_config.yml")
    # todo: handle exception
    # todo: validate yaml structure (known/required keys)
    with open(config_file_path, 'r') as f:
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

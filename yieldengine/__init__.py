import yaml
import os


def get_global_config():
    base_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(base_dir, "yield_engine_config.yml")
    # todo: handle exception
    # todo: validate yaml structure (known/required keys)
    with open(config_file_path, 'r') as f:
        return yaml.safe_load(f)


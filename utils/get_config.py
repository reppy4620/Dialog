import yaml

from .attribute_dict import AttributeDict


def get_config(file_path):
    with open(file_path, 'r') as f:
        _config = yaml.load(f, Loader=yaml.CLoader)
    config = AttributeDict(_config)
    return config

# Utility functions for config and path management
import os
import yaml

def get_config_path(config_name='config.yaml'):
    return os.path.join(os.path.dirname(__file__), '../config', config_name)

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

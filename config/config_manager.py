import os
import json
from typing import List, Dict

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')

class ConfigManager:
    def __init__(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)

    def save_config(self, config: Dict, config_id: str):
        path = os.path.join(CONFIG_DIR, f'{config_id}.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def list_configs(self) -> List[str]:
        return [f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]

    def load_config(self, config_id: str) -> Dict:
        path = os.path.join(CONFIG_DIR, f'{config_id}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

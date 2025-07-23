import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self._validate_config(config)
        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate required configuration sections and fields"""
        required_sections = ['data_source', 'embed_config', 'vector_db']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
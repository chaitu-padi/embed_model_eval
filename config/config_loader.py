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
        required_sections = [
            'data_source',
            'embed_config',
            'vector_db',
            'chunking',
            'retrieval',
            'evaluation'
        ]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
            
        # Validate subsections
        if config['data_source']['type'] == 'pdf' and 'pdf_config' not in config['data_source']:
            raise ValueError("Missing pdf_config for PDF data source")
        elif config['data_source']['type'] == 'oracle' and 'oracle_config' not in config['data_source']:
            raise ValueError("Missing oracle_config for Oracle data source")
            
        # Validate embed_config
        embed_config = config['embed_config']
        if 'models' not in embed_config or not embed_config['models']:
            raise ValueError("No models specified in embed_config")
            
        # Validate dimension reduction settings
        if embed_config.get('dimension_reduction', {}).get('use_pca', False):
            if 'dimension' not in embed_config:
                raise ValueError("Target dimension required when PCA is enabled")
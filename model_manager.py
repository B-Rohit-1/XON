"""
Model Manager - Centralized model management for Xon AI
"""
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from enum import Enum
from ollama_client import ModelProvider

@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    model_id: str
    task_type: str  # chat, code, vision, audio, embedding, etc.
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    source: ModelProvider = ModelProvider.OLLAMA
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_headers: Optional[Dict[str, str]] = field(default_factory=dict)

class ModelManager:
    """Manages AI models and their configurations with support for both local and API-based models"""
    
    def __init__(self, config_path: str = None, api_config: Dict[str, Any] = None):
        """Initialize the model manager with optional config file and API settings
        
        Args:
            config_path: Path to YAML config file
            api_config: Dictionary with API configuration (e.g., base URL, API key)
        """
        self.logger = logging.getLogger("ModelManager")
        self.models: Dict[str, ModelConfig] = {}
        self.default_models: Dict[str, str] = {}  # task_type -> model_id
        self.api_config = api_config or {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_models()
    
    def _load_default_models(self):
        """Load default model configurations with both local and API options"""
        default_models = [
            # Local models
            ModelConfig(
                name="llama3-8b-local",
                model_id="llama3:8b",
                task_type="chat",
                description="Local general purpose chat model",
                parameters={"temperature": 0.7, "top_p": 0.9},
                is_default=True,
                source=ModelProvider.OLLAMA
            ),
            ModelConfig(
                name="codellama-7b-local",
                model_id="codellama:7b",
                task_type="code",
                description="Local code generation model",
                parameters={"temperature": 0.2, "top_p": 0.95},
                is_default=True,
                source=ModelProvider.OLLAMA
            ),
            # API-based models (examples with common providers)
            ModelConfig(
                name="gpt-4",
                model_id="gpt-4",
                task_type="chat",
                description="OpenAI's GPT-4 model",
                parameters={"temperature": 0.7, "max_tokens": 2000},
                source=ModelProvider.OPENAI,
                api_base="https://api.openai.com/v1",
                api_key="your-openai-api-key",
                api_headers={"Content-Type": "application/json"}
            ),
            ModelConfig(
                name="claude-2",
                model_id="claude-2",
                task_type="chat",
                description="Anthropic's Claude 2 model",
                parameters={"temperature": 0.7, "max_tokens": 4000},
                source=ModelProvider.ANTHROPIC,
                api_base="https://api.anthropic.com/v1",
                api_key="your-anthropic-api-key",
                api_headers={"Content-Type": "application/json"}
            )
        ]
        
        # Add all default models
        for model in default_models:
            self.models[model.name] = model
            if model.is_default:
                self.default_models[model.task_type] = model.name
    
    def add_model(self, config: ModelConfig) -> bool:
        """Add or update a model configuration"""
        if not isinstance(config, ModelConfig):
            self.logger.error("Invalid model configuration")
            return False
            
        self.models[config.name] = config
        
        # Update default model for this task type if needed
        if config.is_default:
            self.default_models[config.task_type] = config.model_id
            
        return True
    
    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name
        
        Args:
            name: Name of the model to retrieve
            
        Returns:
            ModelConfig if found, None otherwise
        """
        return self.models.get(name)
        
    def get_model_config_for_request(self, model_name: str) -> Dict[str, Any]:
        """Get complete configuration for making API requests
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model configuration including API details
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
            
        config = {
            "name": model.name,
            "model_id": model.model_id,
            "parameters": model.parameters.copy(),
            "source": model.source
        }
        
        if model.source == ModelSource.API:
            config.update({
                "api_base": model.api_base,
                "api_key": model.api_key or self.api_config.get("api_key"),
                "headers": {
                    **model.api_headers,
                    "Authorization": f"Bearer {model.api_key or self.api_config.get('api_key')}",
                    "Content-Type": "application/json"
                }
            })
            
        return config
    
    def get_models_by_task(self, task_type: str) -> List[ModelConfig]:
        """Get all models for a specific task type"""
        return [m for m in self.models.values() if m.task_type == task_type]
    
    def get_default_model(self, task_type: str) -> Optional[ModelConfig]:
        """Get the default model for a task type"""
        if task_type in self.default_models:
            return self.models.get(self.default_models[task_type])
        return None
    
    def set_default_model(self, model_name: str, task_type: str = None) -> bool:
        """Set the default model for a task type"""
        model = self.get_model(model_name)
        if not model:
            self.logger.error(f"Model not found: {model_name}")
            return False
            
        if task_type and model.task_type != task_type:
            self.logger.error(f"Model {model_name} is not a {task_type} model")
            return False
            
        # Unset previous default for this task type
        current_default = self.default_models.get(model.task_type)
        if current_default and current_default in self.models:
            self.models[current_default].is_default = False
            
        # Set new default
        model.is_default = True
        self.default_models[model.task_type] = model.model_id
        return True
    
    def load_config(self, config_path: str) -> bool:
        """Load model configurations from a YAML file
        
        Example YAML format:
            models:
              - name: gpt-4
                model_id: gpt-4
                task_type: chat
                source: api
                api_base: https://api.openai.com/v1
                api_key: your-api-key
                parameters:
                  temperature: 0.7
                  max_tokens: 2000
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            for model_data in config_data.get('models', []):
                # Convert string source to enum if needed
                if 'source' in model_data and isinstance(model_data['source'], str):
                    model_data['source'] = ModelSource[model_data['source'].upper()]
                model = ModelConfig(**model_data)
                self.add_model(model)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model config: {e}", exc_info=True)
            return False
    
    def save_config(self, config_path: str) -> bool:
        """Save current model configurations to a YAML file
        
        Args:
            config_path: Path to save the configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            config_data = {'models': []}
            
            for model in self.models.values():
                model_data = {
                    'name': model.name,
                    'model_id': model.model_id,
                    'task_type': model.task_type,
                    'description': model.description,
                    'parameters': model.parameters,
                    'is_default': model.is_default,
                    'source': model.source.name.lower()  # Convert enum to string
                }
                
                # Only include API-related fields if it's an API model
                if model.source == ModelSource.API:
                    model_data.update({
                        'api_base': model.api_base,
                        'api_key': model.api_key or '',  # Be careful with saving API keys
                        'api_headers': model.api_headers
                    })
                
                config_data['models'].append(model_data)
            
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_data, f, sort_keys=False, default_flow_style=False)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model config: {e}", exc_info=True)
            return False

# Global model manager instance
model_manager = ModelManager()

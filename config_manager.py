"""
Configuration Manager for XonAI

This module provides a centralized configuration management system that loads settings
from multiple sources with the following priority order:
1. Environment variables (highest priority)
2. config.yaml file
3. Default values (lowest priority)
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_id: str
    task_type: str = "chat"
    description: str = ""
    source: str = "api"
    api_base: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False

@dataclass
class APIConfig:
    """API configuration."""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_host: str = "localhost:11434"

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/app.log"
    max_size: int = 10  # MB
    backup_count: int = 5

@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    ttl: int = 3600  # seconds
    directory: str = ".cache"

@dataclass
class ModelSettings:
    """Model-specific settings."""
    text_model: str = "llama3.2:3b"
    vision_model: str = "llava:7b"
    audio_model: str = "whisper:base"
    max_context_tokens: int = 4096
    memory_enabled: bool = True

@dataclass
class AppConfig:
    """Main application configuration."""
    models: List[ModelConfig] = field(default_factory=list)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    request_timeout: int = 30
    max_retries: int = 3
    default_model: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from a dictionary."""
        # Handle models
        models = []
        for model_data in config_dict.get("models", []):
            models.append(ModelConfig(**model_data))
        
        # Handle API config
        api_config = APIConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_host=os.getenv("OLLAMA_HOST", "localhost:11434")
        )
        
        # Override with config file values if present
        if "api_config" in config_dict:
            api_data = config_dict["api_config"]
            api_config.openai_api_key = api_data.get("openai_api_key", api_config.openai_api_key)
            api_config.anthropic_api_key = api_data.get("anthropic_api_key", api_config.anthropic_api_key)
            api_config.ollama_base_url = api_data.get("ollama_base_url", api_config.ollama_base_url)
            api_config.ollama_host = api_data.get("ollama_host", api_config.ollama_host)
        
        # Create model settings
        model_settings = ModelSettings(
            text_model=os.getenv("TEXT_MODEL", "llama3.2:3b"),
            vision_model=os.getenv("VISION_MODEL", "llava:7b"),
            audio_model=os.getenv("AUDIO_MODEL", "whisper:base"),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4096")),
            memory_enabled=os.getenv("MEMORY_ENABLED", "true").lower() == "true"
        )
        
        # Override with config file values if present
        if "model_settings" in config_dict:
            model_data = config_dict["model_settings"]
            model_settings.text_model = model_data.get("text_model", model_settings.text_model)
            model_settings.vision_model = model_data.get("vision_model", model_settings.vision_model)
            model_settings.audio_model = model_data.get("audio_model", model_settings.audio_model)
            model_settings.max_context_tokens = model_data.get("max_context_tokens", model_settings.max_context_tokens)
            model_settings.memory_enabled = model_data.get("memory_enabled", model_settings.memory_enabled)
        
        # Create config instance
        return cls(
            models=models,
            api=api_config,
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO").upper(),
                file=os.getenv("LOG_FILE", "logs/app.log"),
            ),
            cache=CacheConfig(
                enabled=os.getenv("ENABLE_CACHE", "true").lower() == "true",
                ttl=int(os.getenv("CACHE_TTL", "3600")),
                directory=os.getenv("CACHE_DIR", ".cache")
            ),
            model_settings=model_settings,
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            default_model=os.getenv("DEFAULT_MODEL")
        )

class ConfigManager:
    """Manages application configuration."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._config_path = Path("config.yaml")
        self._config = self._load_config()
        self._initialized = True
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file and environment variables."""
        config_dict = {}
        
        # Load from YAML if exists
        if self._config_path.exists():
            with open(self._config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        
        # Create config object
        return AppConfig.from_dict(config_dict)
    
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self._config
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model by name."""
        for model in self._config.models:
            if model.name == model_name:
                return model
        return None
    
    def get_default_model(self) -> Optional[ModelConfig]:
        """Get the default model configuration."""
        # First try the explicitly set default model
        if self._config.default_model:
            default = self.get_model_config(self._config.default_model)
            if default:
                return default
        
        # Then look for models marked as default
        for model in self._config.models:
            if model.is_default:
                return model
        
        # Return the first model if no default is set
        return self._config.models[0] if self._config.models else None
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        # Update models
        if "models" in new_config:
            self._config.models = [
                ModelConfig(**model_data) 
                for model_data in new_config["models"]
            ]
        
        # Update API config
        if "api_config" in new_config:
            api_data = new_config["api_config"]
            self._config.api = APIConfig(
                openai_api_key=api_data.get("openai_api_key", self._config.api.openai_api_key),
                anthropic_api_key=api_data.get("anthropic_api_key", self._config.api.anthropic_api_key),
                ollama_base_url=api_data.get("ollama_base_url", self._config.api.ollama_base_url)
            )
        
        # Update logging config
        if "logging" in new_config:
            logging_data = new_config["logging"]
            self._config.logging = LoggingConfig(
                level=logging_data.get("level", self._config.logging.level),
                file=logging_data.get("file", self._config.logging.file),
                max_size=logging_data.get("max_size", self._config.logging.max_size),
                backup_count=logging_data.get("backup_count", self._config.logging.backup_count)
            )
        
        # Update cache config
        if "cache" in new_config:
            cache_data = new_config["cache"]
            self._config.cache = CacheConfig(
                enabled=cache_data.get("enabled", self._config.cache.enabled),
                ttl=cache_data.get("ttl", self._config.cache.ttl),
                directory=cache_data.get("directory", self._config.cache.directory)
            )
        
        # Update other settings
        if "request_timeout" in new_config:
            self._config.request_timeout = int(new_config["request_timeout"])
        if "max_retries" in new_config:
            self._config.max_retries = int(new_config["max_retries"])
        if "default_model" in new_config:
            self._config.default_model = new_config["default_model"]
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        config_dict = {
            "models": [
                {
                    "name": model.name,
                    "model_id": model.model_id,
                    "task_type": model.task_type,
                    "description": model.description,
                    "source": model.source,
                    "api_base": model.api_base,
                    "parameters": model.parameters,
                    "is_default": model.is_default
                }
                for model in self._config.models
            ],
            "api_config": {
                "openai_api_key": self._config.api.openai_api_key,
                "anthropic_api_key": self._config.api.anthropic_api_key,
                "ollama_base_url": self._config.api.ollama_base_url
            },
            "logging": {
                "level": self._config.logging.level,
                "file": self._config.logging.file,
                "max_size": self._config.logging.max_size,
                "backup_count": self._config.logging.backup_count
            },
            "cache": {
                "enabled": self._config.cache.enabled,
                "ttl": self._config.cache.ttl,
                "directory": self._config.cache.directory
            },
            "request_timeout": self._config.request_timeout,
            "max_retries": self._config.max_retries,
            "default_model": self._config.default_model
        }
        
        # Ensure directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(self._config_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get the current application configuration."""
    return config_manager.get_config()

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    return config_manager.get_model_config(model_name)

def get_default_model() -> Optional[ModelConfig]:
    """Get the default model configuration."""
    return config_manager.get_default_model()

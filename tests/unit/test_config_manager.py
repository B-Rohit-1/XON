"""Tests for the config_manager module."""
import os
import pytest
from pathlib import Path

def test_config_loading(config_file, sample_config):
    """Test loading configuration from a file."""
    from config_manager import ConfigManager
    
    # Initialize config manager with test config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Verify the config was loaded correctly
    assert config.model_settings.text_model == sample_config["model_settings"]["text_model"]
    assert config.api.openai_api_key == sample_config["api_config"]["openai_api_key"]

def test_default_model_selection(sample_config):
    """Test default model selection logic."""
    from config_manager import ConfigManager
    
    # Create a config manager and update with test config
    config_manager = ConfigManager()
    config_manager.update_config(sample_config)
    
    # Get default model
    default_model = config_manager.get_default_model()
    assert default_model is not None
    assert default_model.name == "test-model"

def test_environment_variable_override():
    """Test that environment variables override config file values."""
    from config_manager import ConfigManager
    
    # Set environment variable
    test_key = "test-env-key-123"
    os.environ["OPENAI_API_KEY"] = test_key
    
    # Initialize config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Verify environment variable took precedence
    assert config.api.openai_api_key == test_key
    
    # Cleanup
    del os.environ["OPENAI_API_KEY"]

def test_config_save_load(tmp_path, sample_config):
    """Test saving and loading configuration."""
    from config_manager import ConfigManager
    
    # Create a test config file
    config_path = tmp_path / "test_config_save.yaml"
    
    # Initialize and save config
    config_manager = ConfigManager()
    config_manager.update_config(sample_config)
    config_manager._config_path = config_path
    config_manager.save_config()
    
    # Verify file was created
    assert config_path.exists()
    
    # Load the saved config
    loaded_config_manager = ConfigManager()
    loaded_config_manager._config_path = config_path
    loaded_config = loaded_config_manager.get_config()
    
    # Verify the loaded config matches the original
    assert loaded_config.model_settings.text_model == sample_config["model_settings"]["text_model"]
    assert loaded_config.api.openai_api_key == sample_config["api_config"]["openai_api_key"]

def test_get_model_config(sample_config):
    """Test getting configuration for a specific model."""
    from config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config_manager.update_config(sample_config)
    
    # Test getting an existing model
    model_config = config_manager.get_model_config("test-model")
    assert model_config is not None
    assert model_config.name == "test-model"
    
    # Test getting a non-existent model
    assert config_manager.get_model_config("non-existent-model") is None

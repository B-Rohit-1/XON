"""Test configuration and fixtures for Xon AI tests."""
import os
import sys
import tempfile
from pathlib import Path
import pytest
from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Setup logging for tests"""
    # Disable log output during tests unless explicitly enabled
    logger.remove()
    logger.add(
        sys.stderr,
        level="ERROR",
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
        temp_log_path = f.name
    
    yield temp_log_path
    
    # Cleanup
    try:
        os.unlink(temp_log_path)
    except:
        pass

@pytest.fixture
def sample_config():
    """Return a sample configuration for testing"""
    return {
        "models": [
            {
                "name": "test-model",
                "model_id": "test-model-1",
                "task_type": "chat",
                "description": "Test model",
                "source": "test",
                "is_default": True
            }
        ],
        "api_config": {
            "openai_api_key": "test-key-123",
            "anthropic_api_key": "test-key-456"
        },
        "model_settings": {
            "text_model": "test-model-1",
            "vision_model": "test-vision-1",
            "audio_model": "test-audio-1"
        }
    }

@pytest.fixture
def config_file(tmp_path, sample_config):
    """Create a temporary config file for testing"""
    import yaml
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(sample_config, f)
    return config_path

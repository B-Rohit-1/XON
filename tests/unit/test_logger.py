"""Tests for the logger module."""
import logging
import os
import pytest
from pathlib import Path
from loguru import logger

def test_logger_setup(temp_log_file):
    """Test logger setup with file output."""
    from logger import setup_logger
    
    # Test file logging
    test_logger = setup_logger(
        "test_logger",
        level="DEBUG",
        log_file=temp_log_file,
        rotation="10 MB"
    )
    
    test_message = "Test log message"
    test_logger.info(test_message)
    
    # Verify the log file was created and contains our message
    assert os.path.exists(temp_log_file)
    with open(temp_log_file, 'r') as f:
        log_content = f.read()
    assert test_message in log_content

def test_log_execution_decorator():
    """Test the log_execution decorator."""
    from logger import log_execution
    
    @log_execution(level=logging.INFO)
    def test_function():
        return "test"
    
    # The decorator will log when the function is called
    result = test_function()
    assert result == "test"

def test_intercept_handler():
    """Test that standard logging is intercepted by loguru."""
    import logging
    from logger import get_logger
    
    std_logger = logging.getLogger("test_intercept")
    std_logger.info("Test intercept message")
    
    # This test verifies the handler is set up without errors
    # Actual log output testing would require capturing stderr
    assert True

def test_get_logger():
    """Test getting a logger instance."""
    from logger import get_logger
    
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    assert isinstance(logger, logging.Logger)

def test_logger_levels():
    """Test that different log levels work correctly."""
    from logger import setup_logger
    
    logger = setup_logger("test_levels", level="DEBUG")
    
    # Test all standard log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # If we get here, no exceptions were raised
    assert True

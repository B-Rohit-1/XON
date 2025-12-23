"""
Advanced logging configuration for Xon AI using loguru

This module provides a robust logging solution with:
- Structured logging with context
- Async support
- File rotation and retention
- Colored console output
- Integration with standard logging
"""
import os
import sys
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar, Callable
from functools import wraps

from loguru import logger as loguru_logger
from loguru._defaults import LOGURU_FORMAT

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and redirect them to loguru"""
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logger(
    name: str = "XonAI",
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    enqueue: bool = True,
    backtrace: bool = True,
    diagnose: bool = False,
    serialize: bool = False,
    catch: bool = True,
) -> logging.Logger:
    """Configure and return a logger instance with enhanced features
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int)
        log_file: Path to log file. If None, logs only to stderr
        rotation: Log rotation size or time (e.g., '10 MB', '1 day')
        retention: Log retention period (e.g., '30 days', '1 month')
        enqueue: Whether to enqueue log messages (thread-safe)
        backtrace: Whether to show the full stack trace
        diagnose: Whether to show variable values in stack traces
        serialize: Whether to output logs as JSON
        catch: Whether to catch and report errors during logging
        
    Returns:
        Configured standard logging.Logger instance
    """
    # Remove default handler
    loguru_logger.remove()
    
    # Configure console handler
    loguru_logger.add(
        sys.stderr,
        level=level.upper() if isinstance(level, str) else level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ) if not serialize else None,
        serialize=serialize,
        enqueue=enqueue,
        backtrace=backtrace,
        diagnose=diagnose,
        catch=catch,
    )
    
    # Add file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            str(log_path),
            rotation=rotation,
            retention=retention,
            enqueue=enqueue,
            level=level.upper() if isinstance(level, str) else level,
            format=LOGURU_FORMAT if not serialize else None,
            serialize=serialize,
            backtrace=backtrace,
            diagnose=diagnose,
            catch=catch,
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set log levels for noisy loggers
    for logger_name in ["urllib3", "httpx", "httpcore", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Return a standard logging logger that works with loguru
    logger = logging.getLogger(name)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with the given name
    
    Args:
        name: Logger name. If None, returns the root logger.
        
    Returns:
        Configured logging.Logger instance
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)

def log_execution(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """Decorator to log function execution
    
    Args:
        logger: Logger instance to use. If None, creates a new one.
        level: Logging level to use
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger_instance = logger or get_logger(func.__module__)
            logger_instance.log(level, f"Starting {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger_instance.log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger_instance.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper  # type: ignore
    return decorator

# Create a default logger instance
logger = get_logger("XonAI")

if __name__ == "__main__":
    # Example usage
    log = setup_logger("test", log_file="logs/test.log")
    log.info("Logger configured successfully!")
    
    try:
        1 / 0
    except Exception as e:
        log.exception("An error occurred")

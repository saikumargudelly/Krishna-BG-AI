import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """
    Set up centralized logging configuration with rotation and proper formatting.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / f"raadhe_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create separate loggers for different components
    loggers = {
        'api': logging.getLogger('api'),
        'cache': logging.getLogger('cache'),
        'model': logging.getLogger('model'),
        'memory': logging.getLogger('memory')
    }
    
    # Set levels for specific loggers
    loggers['api'].setLevel(logging.INFO)
    loggers['cache'].setLevel(logging.DEBUG)
    loggers['model'].setLevel(logging.INFO)
    loggers['memory'].setLevel(logging.DEBUG)
    
    return loggers

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Name of the component (api, cache, model, memory)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 
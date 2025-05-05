import logging
from functools import wraps
from typing import Any, Callable, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')

def safe_execute(func: Callable[..., T], default: Optional[T] = None) -> Optional[T]:
    """
    Safely execute a function and return a default value if it fails.
    
    Args:
        func: The function to execute
        default: The default value to return if the function fails
        
    Returns:
        The result of the function or the default value if it fails
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default

def handle_errors(default: Optional[T] = None):
    """
    Decorator to handle errors in a function.
    
    Args:
        default: The default value to return if the function fails
        
    Returns:
        Decorated function that returns the default value if it fails
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            return safe_execute(lambda: func(*args, **kwargs), default)
        return wrapper
    return decorator 
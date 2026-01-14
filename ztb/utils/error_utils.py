#!/usr/bin/env python3
"""
Error handling utilities for consistent error management across the codebase.
"""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

@contextmanager
def safe_operation_context(operation_name: str, default_result: Any = None):
    """
    Context manager for safe operations with consistent error handling.

    Args:
        operation_name: Name of the operation for logging
        default_result: Default result to return on error
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}")
        if default_result is not None:
            return default_result
        raise

def safe_execute(
    func: Callable[..., T],
    operation_name: str,
    default_result: Optional[T] = None,
    *args: Any,
    **kwargs: Any
) -> Optional[T]:
    """
    Safely execute a function with consistent error handling.

    Args:
        func: Function to execute
        operation_name: Name of the operation for logging
        default_result: Default result to return on error
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default_result on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}")
        return default_result

def log_and_continue(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """
    Decorator to log errors and continue execution.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that logs errors and returns None on failure
    """
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


def safe_operation(
    func: Callable[..., T],
    operation_name: Optional[str] = None,
    default_result: Optional[T] = None,
    collect_errors: bool = False,
    error_list: Optional[list] = None,
) -> Optional[T]:
    """
    Execute a function safely with error handling.
    
    Supports error collection for parallel/multiprocessing scenarios.
    
    Args:
        func: Callable to execute
        operation_name: Name for logging. If None, uses func.__name__
        default_result: Value to return on error
        collect_errors: If True, append errors to error_list instead of raising
        error_list: List to collect errors into (required if collect_errors=True)
    
    Returns:
        Function result or default_result on error
        
    Example:
        # Simple error handling
        result = safe_operation(
            lambda: risky_operation(),
            operation_name="risky_op",
            default_result=None
        )
        
        # Multiprocessing error collection
        errors = []
        for task in tasks:
            result = safe_operation(
                lambda: evaluate_task(task),
                operation_name=f"task_{task.id}",
                collect_errors=True,
                error_list=errors,
            )
    """
    op_name = operation_name or (getattr(func, '__name__', 'unknown'))
    
    try:
        return func()
    except Exception as e:
        logger.error(f"Error in {op_name}: {e}", exc_info=True)
        
        if collect_errors:
            if error_list is not None:
                error_list.append({
                    'operation': op_name,
                    'error': str(e),
                    'error_type': type(e).__name__,
                })
            return default_result
        else:
            return default_result
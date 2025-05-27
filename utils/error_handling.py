"""
Error handling utilities for the subtitle generator application.
"""
import functools
import logging
import traceback
from typing import Callable, TypeVar, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('subtitle_generator')

# Type variable for function annotations
T = TypeVar('T')


class ProcessingError(Exception):
    """Base exception for all application-specific errors."""
    pass


class FileProcessingError(ProcessingError):
    """Exception raised for errors during file processing."""
    pass


class TranscriptionError(ProcessingError):
    """Exception raised for errors during audio transcription."""
    pass


class SubtitleError(ProcessingError):
    """Exception raised for errors during subtitle generation or processing."""
    pass


class TranslationError(ProcessingError):
    """Exception raised for errors during translation."""
    pass


class SummarizationError(ProcessingError):
    """Exception raised for errors during content summarization."""
    pass


class SpeakerRecognitionError(ProcessingError):
    """Exception raised for errors during speaker recognition."""
    pass


def handle_errors(
    default_return: Optional[Any] = None, 
    log_exception: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for handling errors in functions.
    
    Args:
        default_return: Value to return if an exception occurs
        log_exception: Whether to log the exception
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(traceback.format_exc())
                
                # Re-raise specific application errors for proper handling
                if isinstance(e, ProcessingError):
                    raise
                
                # For other exceptions, wrap in an appropriate application error
                if 'file' in func.__name__.lower():
                    raise FileProcessingError(f"File processing error: {str(e)}") from e
                elif 'transcri' in func.__name__.lower():
                    raise TranscriptionError(f"Transcription error: {str(e)}") from e
                elif 'subtitle' in func.__name__.lower():
                    raise SubtitleError(f"Subtitle error: {str(e)}") from e
                elif 'translat' in func.__name__.lower():
                    raise TranslationError(f"Translation error: {str(e)}") from e
                elif 'summar' in func.__name__.lower():
                    raise SummarizationError(f"Summarization error: {str(e)}") from e
                elif 'speaker' in func.__name__.lower():
                    raise SpeakerRecognitionError(f"Speaker recognition error: {str(e)}") from e
                else:
                    raise ProcessingError(f"Processing error: {str(e)}") from e
                
        return wrapper
    return decorator


def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Completed {func.__name__}")
        return result
    
    return wrapper
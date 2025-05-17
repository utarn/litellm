"""
Utility functions for sanitizing error responses to hide sensitive details from clients
while preserving full logging for debugging.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import status
from litellm.proxy._types import ProxyException

verbose_proxy_logger = logging.getLogger("litellm.proxy")


def get_sanitized_error_message(exception: Exception, exception_type: str = "APIConnectionError") -> str:
    """
    Returns a sanitized error message that hides implementation details from clients.
    
    Args:
        exception: The original exception
        exception_type: The type of exception for client response
        
    Returns:
        A sanitized error message appropriate for client consumption
    """
    # Map common error types to sanitized messages
    sanitized_messages = {
        "APIConnectionError": "Service temporarily unavailable. Please try again later.",
        "OpenAIError": "Request processing failed. Please check your request and try again.",
        "AuthenticationError": "Invalid authentication credentials.",
        "RateLimitError": "Rate limit exceeded. Please wait before making more requests.",
        "BadRequestError": "Invalid request format. Please check your request parameters.",
        "TimeoutError": "Request timed out. Please try again.",
        "InternalServerError": "Internal server error. Please try again later.",
        "ValidationError": "Request validation failed. Please check your request parameters.",
    }
    
    # Return sanitized message or a generic fallback
    return sanitized_messages.get(exception_type, "An error occurred while processing your request. Please try again.")


def sanitize_error_response(
    exception: Exception,
    status_code: Optional[int] = None,
    exception_type: Optional[str] = None,
    preserve_auth_errors: bool = True
) -> Dict[str, Any]:
    """
    Creates a sanitized error response that hides sensitive implementation details
    while preserving the essential error information for the client.
    
    Args:
        exception: The original exception
        status_code: HTTP status code (will be determined from exception if not provided)
        exception_type: Exception type for client (will be determined from exception if not provided)
        preserve_auth_errors: Whether to preserve original auth error messages
        
    Returns:
        Dictionary containing sanitized error response
    """
    # Log the full error details for debugging
    verbose_proxy_logger.error(
        f"Exception occurred: {type(exception).__name__}: {str(exception)}",
        exc_info=True
    )
    
    # Determine status code
    if status_code is None:
        if hasattr(exception, 'status_code') and isinstance(getattr(exception, 'status_code'), int):
            status_code = getattr(exception, 'status_code')
        elif hasattr(exception, 'code') and isinstance(getattr(exception, 'code'), int):
            status_code = getattr(exception, 'code')
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Determine exception type for client
    if exception_type is None:
        if hasattr(exception, 'type') and isinstance(getattr(exception, 'type'), str):
            exception_type = getattr(exception, 'type')
        else:
            exception_type = type(exception).__name__
    
    # Ensure exception_type is not None
    if exception_type is None:
        exception_type = "InternalServerError"
    
    # Preserve authentication errors as they need to be specific
    if preserve_auth_errors and (
        "auth" in str(exception_type).lower() or
        status_code in [401, 403] or
        "authentication" in str(exception).lower() or
        "authorization" in str(exception).lower()
    ):
        message = getattr(exception, 'message', str(exception))
    else:
        # Use sanitized message for other errors
        message = get_sanitized_error_message(exception, exception_type)
    
    # Create sanitized response
    sanitized_response = {
        "error": {
            "message": message,
            "type": exception_type,
            "code": status_code
        }
    }
    
    # Add param if it's safe to expose (not for internal errors)
    if hasattr(exception, 'param') and status_code is not None and status_code < 500:
        sanitized_response["error"]["param"] = getattr(exception, 'param')
    
    return sanitized_response


def create_sanitized_proxy_exception(
    original_exception: Exception,
    fallback_message: str = "An error occurred while processing your request",
    preserve_auth_errors: bool = True
) -> ProxyException:
    """
    Creates a sanitized ProxyException from an original exception.
    
    Args:
        original_exception: The original exception
        fallback_message: Fallback message if sanitization fails
        preserve_auth_errors: Whether to preserve original auth error messages
        
    Returns:
        A sanitized ProxyException
    """
    try:
        # Log the original exception with full details
        verbose_proxy_logger.error(
            f"Creating sanitized exception from: {type(original_exception).__name__}: {str(original_exception)}",
            exc_info=True
        )
        
        # Get sanitized response
        sanitized = sanitize_error_response(original_exception, preserve_auth_errors=preserve_auth_errors)
        error_info = sanitized["error"]
        
        return ProxyException(
            message=error_info["message"],
            type=error_info["type"],
            param=error_info.get("param", "None"),
            code=error_info["code"]
        )
        
    except Exception as sanitization_error:
        # If sanitization itself fails, log and return a safe fallback
        verbose_proxy_logger.error(
            f"Error during sanitization: {sanitization_error}. Original error: {original_exception}",
            exc_info=True
        )
        
        return ProxyException(
            message=fallback_message,
            type="InternalServerError",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
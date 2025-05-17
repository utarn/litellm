import pytest
import httpx
from unittest.mock import Mock

from litellm.litellm_core_utils.exception_mapping_utils import exception_type
import litellm


def test_exception_mapping_with_generic_exception_no_request_attribute():
    """
    Test that exception_type handles generic Exception objects that don't have a request attribute.
    
    This tests the fix for the bug: AttributeError: 'Exception' object has no attribute 'request'
    """
    # Create a generic exception without a request attribute
    original_exception = Exception("Test error message")
    
    # Ensure the exception doesn't have a request attribute
    assert not hasattr(original_exception, "request")
    
    # This should not raise AttributeError anymore
    try:
        exception_type(
            model="test-model",
            original_exception=original_exception,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={}
        )
    except AttributeError as e:
        if "'Exception' object has no attribute 'request'" in str(e):
            pytest.fail(f"The bug still exists: {e}")
        else:
            # Re-raise if it's a different AttributeError
            raise
    except Exception:
        # Other exceptions are expected (like APIConnectionError), we just want to ensure
        # no AttributeError about missing 'request' attribute
        pass


def test_exception_mapping_with_request_attribute_still_works():
    """
    Test that exception_type still works correctly when the exception has a request attribute.
    """
    # Create a mock exception with a request attribute
    original_exception = Exception("Test error message with request")
    original_exception.request = httpx.Request(method="POST", url="https://api.openai.com/v1/chat/completions")
    original_exception.status_code = 401
    
    # Ensure the exception has a request attribute
    assert hasattr(original_exception, "request")
    
    # This should work and use the request attribute
    try:
        exception_type(
            model="test-model",
            original_exception=original_exception,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={}
        )
    except litellm.AuthenticationError as e:
        # This is expected for a 401 error
        pass
    except Exception as e:
        # Should not raise AttributeError about missing request
        if "object has no attribute 'request'" in str(e):
            pytest.fail(f"Unexpected AttributeError: {e}")


def test_exception_mapping_with_none_request_attribute():
    """
    Test that exception_type handles exceptions where request attribute is None.
    """
    # Create a mock exception with None request attribute
    original_exception = Exception("Test error message with None request")
    original_exception.request = None
    original_exception.status_code = 500
    
    # This should work and handle None request gracefully
    try:
        exception_type(
            model="test-model",
            original_exception=original_exception,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={}
        )
    except Exception as e:
        # Should not raise AttributeError
        if "object has no attribute 'request'" in str(e):
            pytest.fail(f"Unexpected AttributeError: {e}")


def test_exception_mapping_various_providers_no_request():
    """
    Test that the fix works for different providers that might create APIError exceptions.
    """
    providers_to_test = [
        "cohere",
        "huggingface", 
        "ai21",
        "nlp_cloud",
        "together_ai",
        "vllm",
        "openrouter"
    ]
    
    for provider in providers_to_test:
        # Create a generic exception without request attribute but with status_code
        original_exception = Exception(f"Test error for {provider}")
        original_exception.status_code = 429  # This triggers APIError creation
        original_exception.message = f"Rate limit error for {provider}"
        
        # Ensure no request attribute
        assert not hasattr(original_exception, "request")
        
        try:
            exception_type(
                model="test-model",
                original_exception=original_exception,
                custom_llm_provider=provider,
                completion_kwargs={},
                extra_kwargs={}
            )
        except AttributeError as e:
            if "'Exception' object has no attribute 'request'" in str(e):
                pytest.fail(f"The bug still exists for provider {provider}: {e}")
            else:
                raise
        except Exception:
            # Other exceptions are expected
            pass
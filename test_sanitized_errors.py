#!/usr/bin/env python3
"""
Test script to verify that error sanitization works correctly.
This test simulates the APIConnectionError scenario from the task description.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from litellm.proxy.sanitize_error_responses import sanitize_error_response
from litellm import APIConnectionError

def test_openrouter_error_sanitization():
    """Test that detailed OpenRouter errors are sanitized for client response"""
    
    # Simulate the error from the task description
    detailed_error_message = (
        "APIConnectionError: OpenrouterException - openrouter raised a streaming error - "
        "finish_reason: error, no content string given. Received Chunk={'text': '', "
        "'is_finished': True, 'finish_reason': 'error', 'logprobs': None, "
        "'original_chunk': ModelResponseStream(id='gen-1753442092-Wd4RnwiBbrSDteejLCWk', "
        "created=1753442092, model='qwen/qwen3-coder:free', object='chat.completion.chunk', "
        "system_fingerprint=None, choices=[StreamingChoices(error={'message': "
        "'Error processing stream', 'code': 502, 'metadata': "
        "{'provider_name': 'Chutes', 'raw': {'retryable': True}}}, "
        "native_finish_reason='error', finish_reason='error', index=0, "
        "delta=Delta(provider_specific_fields=None, content='', role='assistant', "
        "function_call=None, tool_calls=None, audio=None), logprobs=None)], "
        "provider_specific_fields=None, usage=None), 'usage': None}"
    )
    
    # Create an APIConnectionError with the detailed message
    api_error = APIConnectionError(
        message=detailed_error_message,
        llm_provider="openrouter",
        model="qwen/qwen3-coder:free"
    )
    
    # Test sanitization
    sanitized = sanitize_error_response(api_error)
    
    print("=== Original Error (would be logged) ===")
    print(f"Type: {type(api_error).__name__}")
    print(f"Message: {detailed_error_message}")
    print()
    
    print("=== Sanitized Response (sent to client) ===")
    print(f"Content: {sanitized}")
    print()
    
    # Verify that detailed information is not in the sanitized response
    assert "gen-1753442092-Wd4RnwiBbrSDteejLCWk" not in str(sanitized), "Request ID should not be exposed"
    assert "ModelResponseStream" not in str(sanitized), "Internal types should not be exposed"
    assert "provider_name" not in str(sanitized), "Provider details should not be exposed"
    assert "Chutes" not in str(sanitized), "Internal provider names should not be exposed"
    
    # Verify that basic error information is still present
    assert "error" in sanitized, "Error structure should be present"
    assert sanitized["error"]["type"] == "APIConnectionError", "Error type should be preserved"
    
    print("✅ Test passed: Detailed error information is properly sanitized")
    return True

def test_generic_exception_sanitization():
    """Test that generic exceptions are also sanitized"""
    
    # Create a generic exception with sensitive information
    sensitive_error = Exception("Database connection failed: host=internal.db.company.com, user=admin, password=secret123")
    
    sanitized = sanitize_error_response(sensitive_error)
    
    print("=== Generic Exception Test ===")
    print(f"Original: {str(sensitive_error)}")
    print(f"Sanitized: {sanitized}")
    print()
    
    # Verify sensitive info is not exposed
    assert "internal.db.company.com" not in str(sanitized), "Internal hostnames should not be exposed"
    assert "password=secret123" not in str(sanitized), "Passwords should not be exposed"
    assert "admin" not in str(sanitized), "Usernames should not be exposed"
    
    print("✅ Generic exception sanitization test passed")
    return True

if __name__ == "__main__":
    print("Testing Error Sanitization System")
    print("=" * 50)
    
    try:
        test_openrouter_error_sanitization()
        test_generic_exception_sanitization()
        print("\n🎉 All tests passed! Error sanitization is working correctly.")
        print("\nKey benefits:")
        print("- Detailed error information is logged but not sent to clients")
        print("- Only safe, generic error messages are returned to users")
        print("- HTTP status codes and error types are preserved for client handling")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
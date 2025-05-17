#!/usr/bin/env python3
"""
Test script to verify ModelHarbor headers are applied correctly to OpenAI and OpenRouter requests only.
"""

import sys
import os

# Add the current directory to path to import litellm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from litellm.llms.custom_httpx.http_handler import get_modelharbor_headers
    from litellm._version import version
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run this script from the litellm root directory")
    sys.exit(1)

def test_get_modelharbor_headers():
    """Test that ModelHarbor headers are returned only for OpenAI and OpenRouter providers."""
    
    print("Testing get_modelharbor_headers function...")
    
    # Test OpenAI provider
    openai_headers = get_modelharbor_headers("openai")
    expected_headers = {
        "HTTP-Referer": "https://github.com/ModelHarbor/ModelHarbor-Agent",
        "X-Title": "ModelHarbor Agent",
        "User-Agent": f"ModelHarbor/{version}",
    }
    
    print(f"OpenAI headers: {openai_headers}")
    assert openai_headers == expected_headers, f"Expected {expected_headers}, got {openai_headers}"
    print("✓ OpenAI provider returns correct headers")
    
    # Test OpenRouter provider
    openrouter_headers = get_modelharbor_headers("openrouter")
    print(f"OpenRouter headers: {openrouter_headers}")
    assert openrouter_headers == expected_headers, f"Expected {expected_headers}, got {openrouter_headers}"
    print("✓ OpenRouter provider returns correct headers")
    
    # Test other providers (should return empty dict)
    test_providers = ["anthropic", "cohere", "huggingface", "vertex_ai", "azure", "bedrock", "replicate"]
    
    for provider in test_providers:
        headers = get_modelharbor_headers(provider)
        print(f"{provider} headers: {headers}")
        assert headers == {}, f"Expected empty dict for {provider}, got {headers}"
        print(f"✓ {provider} provider returns no headers (correct)")
    
    # Test None provider
    none_headers = get_modelharbor_headers(None)
    print(f"None provider headers: {none_headers}")
    assert none_headers == {}, f"Expected empty dict for None provider, got {none_headers}"
    print("✓ None provider returns no headers (correct)")
    
    print("\n🎉 All tests passed! ModelHarbor headers are correctly applied only to OpenAI and OpenRouter providers.")

if __name__ == "__main__":
    test_get_modelharbor_headers()
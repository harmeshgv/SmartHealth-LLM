import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json # Import json
from app.llm.adapters.ollama_adapter import OllamaLLM

@pytest.mark.asyncio
@patch('app.llm.adapters.ollama_adapter.httpx.AsyncClient')
async def test_ollama_llm_generate(mock_async_client):
    """
    Tests the generate method of the OllamaLLM adapter with a mocked httpx.AsyncClient.
    """
    # Configure the mock AsyncClient instance
    mock_client_instance = MagicMock()
    mock_async_client.return_value.__aenter__.return_value = mock_client_instance
    
    # Configure the mock response object from client.post
    mock_response = MagicMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response) # Make post awaitable
    
    # Configure the mock response's aiter_lines to yield lines
    async def mock_aiter_lines():
        yield json.dumps({"model": "llama3", "response": "Ollama", "done": False})
        yield json.dumps({"model": "llama3", "response": " test successful.", "done": True})
    
    mock_response.aiter_lines = mock_aiter_lines
    
    llm = OllamaLLM(model="llama3.2:latest", host="http://dummy-host:11434")

    prompt = "Test prompt"
    response = await llm.generate(prompt)

    # Assert that httpx.AsyncClient was instantiated
    mock_async_client.assert_called_once()
    
    # Assert that the post method was called correctly
    mock_client_instance.post.assert_awaited_once_with(
        "http://dummy-host:11434/api/generate",
        json={"model": "llama3.2:latest", "prompt": prompt},
        timeout=None,
    )

    # Assert that the response from the mock is correctly processed
    assert response == "Ollama test successful."

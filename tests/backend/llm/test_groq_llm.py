import pytest
from unittest.mock import patch, MagicMock
from app.llm.adapters.groq_adapter import GroqLLM
from app.config import settings # Import settings


@pytest.fixture
def mock_groq_client():
    """Mocks the groq.Groq client."""
    with patch('app.llm.adapters.groq_adapter.Groq') as MockGroq:
        mock_client = MockGroq.return_value
        
        # Mock the response from the chat completions create method
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "LLM test successful."
        
        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client

@pytest.mark.asyncio
async def test_groq_llm_generate(mock_groq_client):
    """
    Tests the generate method of the GroqLLM adapter with a mocked client.
    """
    # No need to pass api_key here, as it's mocked via settings or the client itself
    llm = GroqLLM(model="llama-3.1-8b-instant")

    prompt = "Test prompt"
    response = await llm.generate(prompt)

    # Assert that the client's create method was called correctly
    mock_groq_client.chat.completions.create.assert_called_once_with(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0 # default value from GroqLLM
    )

    # Assert that the response from the mock is returned
    assert response == "LLM test successful."

def test_groq_llm_initialization_no_api_key(monkeypatch):
    """
    Tests that the GroqLLM adapter raises an error if no API key is provided
    when the real Groq client would do so.
    """
    monkeypatch.setattr(settings, 'GROQ_API_KEY', None)
    
    with patch('app.llm.adapters.groq_adapter.Groq') as MockGroqClient:
        # Configure the mock Groq client to raise a ValueError during initialization
        # if api_key is None, simulating the real client's behavior.
        MockGroqClient.side_effect = ValueError("Groq API key is required")
        
        with pytest.raises(ValueError, match="Groq API key is required"):
            GroqLLM(model="llama-3.1-8b-instant")



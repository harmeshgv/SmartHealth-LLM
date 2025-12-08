import pytest
from app.llm.adapters.ollama_adapter import OllamaLLM

@pytest.mark.asyncio
async def test_ollama_llm():
    llm = OllamaLLM(model="llama3.2:latest")

    result = await llm.generate("Reply with exactly: OK.")

    assert isinstance(result, str)
    assert len(result) > 0
    assert "ok" in result.lower()

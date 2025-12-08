import pytest
from app.llm.adapters.groq_adapter import GroqLLM
from app.config import settings


@pytest.mark.asyncio
async def test_groq_llm():
    llm = GroqLLM(model="llama-3.1-8b-instant")

    prompt = "Reply exactly with: LLM test successful."
    response = await llm.generate(prompt)

    print("\n=== Groq LLM RESPONSE ===")
    print(response)
    print("==========================\n")

    # Assertion: model must reply with the keyword
    assert "LLM test successful" in response

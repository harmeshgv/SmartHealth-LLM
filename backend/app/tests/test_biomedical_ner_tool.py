import pytest
from app.tools.medical.biomedical_ner_tool import BiomedicalNERTool

@pytest.mark.asyncio
async def test_biomedical_ner_tool_basic():
    tool = BiomedicalNERTool()

    text = "The patient reports fever and headache for the last two days."

    entities = await tool.run(text)

    # Output must be a list
    assert isinstance(entities, list)

    # Should extract at least one symptom-like entity
    assert any(sym.lower() in ["fever", "headache"] for sym in entities)


@pytest.mark.asyncio
async def test_biomedical_ner_empty():
    tool = BiomedicalNERTool()

    text = ""
    entities = await tool.run(text)

    # Should return empty list, not crash
    assert isinstance(entities, list)
    assert len(entities) == 0

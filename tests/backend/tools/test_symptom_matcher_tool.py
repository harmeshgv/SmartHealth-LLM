# tests/backend/tools/test_symptom_matcher_tool.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import logging
import pytest_asyncio

from backend.tools.symptom_matcher_tool import SymptomDiseaseMatcherTool
from backend.config import settings
from langchain_core.documents import Document

@pytest.fixture(autouse=True)
def cap_log(caplog):
    caplog.set_level(logging.WARNING)

@pytest.fixture
def mock_embeddings():
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 1536
    return mock_embed

@pytest.fixture
def mock_faiss_vectorstore():
    with patch('backend.tools.symptom_matcher_tool.FAISS.load_local') as mock_load_local:
        mock_vectorstore_instance = MagicMock()
        
        mock_vectorstore_instance.similarity_search_with_score.return_value = [
            (Document(page_content="Fever, cough, sore throat", metadata={"disease": "Common Cold"}), 0.95),
            (Document(page_content="Headache, nausea, sensitivity to light", metadata={"disease": "Migraine"}), 0.88),
        ]
        mock_load_local.return_value = mock_vectorstore_instance
        yield mock_load_local

@pytest.fixture
def symptom_matcher_tool_instance(mock_faiss_vectorstore, mock_embeddings):
    return SymptomDiseaseMatcherTool(
        db_path=settings.SYMPTOM_FAISS_DB,
        embeddings=mock_embeddings
    )

def test_tool_initialization(symptom_matcher_tool_instance, mock_faiss_vectorstore):
    assert symptom_matcher_tool_instance.name == "Symptom to Disease Matcher"
    assert "Finds potential diseases that match a given list of symptoms" in symptom_matcher_tool_instance.description
    mock_faiss_vectorstore.assert_called_once_with(
        settings.SYMPTOM_FAISS_DB,
        ANY,
        allow_dangerous_deserialization=True
    )
    assert hasattr(symptom_matcher_tool_instance, 'vectorstore')

def test_tool_initialization_exception(mock_embeddings):
    with patch('backend.tools.symptom_matcher_tool.FAISS.load_local', side_effect=Exception("FAISS load error")):
        with pytest.raises(Exception, match="FAISS load error"):
            SymptomDiseaseMatcherTool(db_path="invalid_path", embeddings=mock_embeddings)

@pytest.mark.asyncio
async def test_execute_with_valid_symptoms(symptom_matcher_tool_instance, mock_faiss_vectorstore):
    symptoms = ["fever", "cough"]
    result = await symptom_matcher_tool_instance.execute(symptoms=symptoms) # Await the async method

    mock_faiss_vectorstore.return_value.similarity_search_with_score.assert_called_once_with(
        "fever, cough", k=3
    )
    assert "matched_diseases" in result
    assert isinstance(result["matched_diseases"], list)
    assert len(result["matched_diseases"]) == 2
    assert result["matched_diseases"][0]["disease"] == "Common Cold"
    assert result["matched_diseases"][0]["score"] == 0.95
    assert "error" not in result

@pytest.mark.asyncio
async def test_execute_with_empty_symptoms(symptom_matcher_tool_instance):
    result = await symptom_matcher_tool_instance.execute(symptoms=[]) # Await the async method
    assert "error" in result
    assert "non-empty list of symptoms" in result["error"]

@pytest.mark.asyncio
async def test_execute_with_non_list_input(symptom_matcher_tool_instance):
    result = await symptom_matcher_tool_instance.execute(symptoms="fever, cough") # Await the async method
    assert "error" in result
    assert "non-empty list of symptoms" in result["error"]

@pytest.mark.asyncio
async def test_execute_similarity_search_exception(symptom_matcher_tool_instance, mock_faiss_vectorstore):
    mock_faiss_vectorstore.return_value.similarity_search_with_score.side_effect = Exception("Search failed")
    symptoms = ["headache"]
    result = await symptom_matcher_tool_instance.execute(symptoms=symptoms) # Await the async method

    assert "error" in result
    assert "Search failed" in result["error"]
    assert "matched_diseases" not in result

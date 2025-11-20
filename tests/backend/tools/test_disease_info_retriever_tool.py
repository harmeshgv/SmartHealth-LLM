# tests/backend/tools/test_disease_info_retriever_tool.py
import pytest
from unittest.mock import patch, MagicMock
import logging
import pytest_asyncio

from backend.tools.disease_info_retriever_tool import DiseaseInfoRetrieverTool
from backend.config import settings
from langchain_core.documents import Document

@pytest.fixture(autouse=True)
def cap_log(caplog):
    caplog.set_level(logging.WARNING)

@pytest.fixture
def mock_csv_data():
    return [
        {"disease": "Common Cold", "Overview": "Viral infection..."},
        {"disease": "Influenza", "Overview": "Flu virus..."},
        {"disease": "Migraine", "Overview": "Severe headache..."},
    ]

@pytest.fixture
def mock_load_csv(mock_csv_data):
    with patch('backend.tools.disease_info_retriever_tool.DiseaseInfoRetrieverTool._load_csv') as mock_method:
        mock_method.return_value = mock_csv_data
        yield mock_method

@pytest.fixture
def mock_embeddings():
    return MagicMock()

@pytest.fixture
def mock_faiss_from_documents():
    # Correct path for patching
    with patch('langchain_community.vectorstores.FAISS.from_documents') as mock_faiss:
        mock_vectorstore_instance = MagicMock()
        mock_faiss.return_value = mock_vectorstore_instance
        yield mock_faiss

@pytest.fixture
def retriever_tool_instance(mock_load_csv, mock_embeddings, mock_faiss_from_documents):
    return DiseaseInfoRetrieverTool(csv_path=settings.MAYO_CSV, embeddings=mock_embeddings)

def test_tool_initialization(retriever_tool_instance, mock_load_csv, mock_faiss_from_documents):
    assert retriever_tool_instance.name == "Disease Information Retriever"
    mock_load_csv.assert_called_once()
    mock_faiss_from_documents.assert_called_once()
    assert len(retriever_tool_instance.db) == 3
    assert "common cold" in retriever_tool_instance.db_map

@pytest.mark.asyncio
async def test_execute_exact_match(retriever_tool_instance):
    result = await retriever_tool_instance.execute(disease_name="Influenza")
    assert "info" in result
    assert result["info"]["disease"] == "Influenza"
    retriever_tool_instance.name_vectorstore.similarity_search_with_score.assert_not_called()

@pytest.mark.asyncio
async def test_execute_semantic_match(retriever_tool_instance):
    query = "severe head pain"
    mock_doc = Document(page_content="migraine")
    retriever_tool_instance.name_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    
    result = await retriever_tool_instance.execute(disease_name=query)
    
    assert "info" in result
    assert result["info"]["disease"] == "Migraine"
    retriever_tool_instance.name_vectorstore.similarity_search_with_score.assert_called_once_with(query.lower(), k=1, score_threshold=0.85)

@pytest.mark.asyncio
async def test_execute_no_match(retriever_tool_instance):
    query = "Unknown Condition"
    retriever_tool_instance.name_vectorstore.similarity_search_with_score.return_value = []
    
    result = await retriever_tool_instance.execute(disease_name=query)
    
    assert "error" in result
    assert f"Could not find a match for '{query}' in the database." in result["error"]
    retriever_tool_instance.name_vectorstore.similarity_search_with_score.assert_called_once_with(query.lower(), k=1, score_threshold=0.85)

def test_tool_initialization_file_not_found(mock_embeddings):
    with patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            DiseaseInfoRetrieverTool(csv_path="non_existent.csv", embeddings=mock_embeddings)

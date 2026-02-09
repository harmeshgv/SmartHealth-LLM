# tests/backend/tools/test_disease_info_retriever_tool.py
import pytest
from unittest.mock import patch, MagicMock
import logging
import pytest_asyncio

from app.tools.medical.disease_info_tool import DiseaseInfoRetrieverTool
from app.config import settings
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
    with patch('app.tools.medical.disease_info_tool.DiseaseInfoRetrieverTool._load_csv') as mock_method:
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
    with patch('faiss.read_index') as mock_read_index, \
         patch('app.utils.embeddings.EmbeddingSingleton.get_instance') as mock_get_embedding_instance:
        mock_read_index.return_value = MagicMock()
        mock_get_embedding_instance.return_value = mock_embeddings
        return DiseaseInfoRetrieverTool()

def test_tool_initialization(retriever_tool_instance, mock_load_csv, mock_faiss_from_documents):
    assert retriever_tool_instance.name == "disease_info_retriever"
    mock_load_csv.assert_called_once()
    assert retriever_tool_instance.db_map is not None


@pytest.mark.asyncio
async def test_run_exact_match(retriever_tool_instance):
    with patch.object(retriever_tool_instance, '_find_best_match', return_value='influenza') as mock_find:
        result = await retriever_tool_instance.run(disease_name="Influenza")
        assert "info" in result
        assert result["info"]["disease"] == "Influenza"
        mock_find.assert_called_once_with("Influenza")


@pytest.mark.asyncio
async def test_run_semantic_match(retriever_tool_instance):
    query = "severe head pain"
    with patch.object(retriever_tool_instance, '_find_best_match', return_value='migraine') as mock_find:
        result = await retriever_tool_instance.run(disease_name=query)
        assert "info" in result
        assert result["info"]["disease"] == "Migraine"
        mock_find.assert_called_once_with(query)


@pytest.mark.asyncio
async def test_run_no_match(retriever_tool_instance):
    query = "Unknown Condition"
    with patch.object(retriever_tool_instance, '_find_best_match', return_value=None) as mock_find:
        result = await retriever_tool_instance.run(disease_name=query)
        assert "error" in result
        assert f"No match found for {query}" in result["error"]
        mock_find.assert_called_once_with(query)


def test_tool_initialization_file_not_found(mock_embeddings):
    with patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            DiseaseInfoRetrieverTool()

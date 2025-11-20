# tests/backend/tools/test_biomedical_ner_tool.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import logging
import numpy as np
import pytest_asyncio # Import pytest_asyncio

from backend.tools.biomedical_ner_tool import BiomedicalNERTool
from backend.config import settings

@pytest.fixture(autouse=True)
def cap_log(caplog):
    caplog.set_level(logging.WARNING)

@pytest.fixture
def mock_transformers_pipeline_successful():
    with patch('backend.tools.biomedical_ner_tool.pipeline') as mock_pipeline_func:
        mock_pipeline_func.return_value = MagicMock(
            return_value=[
                {'entity': 'B-DISEASE', 'score': np.float32(0.99), 'word': 'diabetes', 'start': 26, 'end': 34, 'entity_group': 'DISEASE'},
                {'entity': 'B-DRUG', 'score': np.float32(0.98), 'word': 'metformin', 'start': 49, 'end': 58, 'entity_group': 'DRUG'}
            ]
        )
        yield mock_pipeline_func

@pytest.fixture
def mock_transformers_pipeline_empty():
    with patch('backend.tools.biomedical_ner_tool.pipeline') as mock_pipeline_func:
        mock_pipeline_func.return_value = MagicMock(return_value=[])
        yield mock_pipeline_func

@pytest.fixture
def mock_transformers_pipeline_exception():
    with patch('backend.tools.biomedical_ner_tool.pipeline') as mock_pipeline_func:
        mock_pipeline_func.return_value = MagicMock(side_effect=Exception("Mock pipeline error"))
        yield mock_pipeline_func

@pytest.fixture
def mock_auto_tokenizer():
    with patch('backend.tools.biomedical_ner_tool.AutoTokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        yield mock_tokenizer

@pytest.fixture
def mock_auto_model_for_token_classification():
    with patch('backend.tools.biomedical_ner_tool.AutoModelForTokenClassification.from_pretrained') as mock_model:
        mock_model.return_value = MagicMock()
        yield mock_model

@pytest.fixture
def ner_tool_instance(mock_auto_tokenizer, mock_auto_model_for_token_classification, mock_transformers_pipeline_successful):
    return BiomedicalNERTool(model_name=settings.BIOMEDICAL_NER_MODEL_NAME)


def test_tool_initialization(ner_tool_instance, mock_auto_tokenizer, mock_auto_model_for_token_classification, mock_transformers_pipeline_successful):
    assert ner_tool_instance.name == "Biomedical NER"
    assert "Extracts biomedical entities" in ner_tool_instance.description
    
    mock_auto_tokenizer.assert_called_once_with(settings.BIOMEDICAL_NER_MODEL_NAME, cache_dir=ANY)
    mock_auto_model_for_token_classification.assert_called_once_with(settings.BIOMEDICAL_NER_MODEL_NAME, cache_dir=ANY)
    assert hasattr(ner_tool_instance, 'pipe')
    mock_transformers_pipeline_successful.assert_called_once_with(
        "ner",
        model=mock_auto_model_for_token_classification.return_value,
        tokenizer=mock_auto_tokenizer.return_value,
        aggregation_strategy="average",
    )

@pytest.mark.asyncio
async def test_execute_with_valid_text_successful(ner_tool_instance, mock_transformers_pipeline_successful):
    text = "The patient was diagnosed with diabetes and prescribed metformin."
    result = await ner_tool_instance.execute(text=text) # Await the async method

    ner_tool_instance.pipe.assert_called_once_with(text)
    
    assert "entities" in result
    assert isinstance(result["entities"], list)
    assert len(result["entities"]) == 2
    assert result["entities"][0]["word"] == "diabetes"
    assert result["entities"][1]["word"] == "metformin"
    assert isinstance(result["entities"][0]["score"], float)
    assert "error" not in result

@pytest.mark.asyncio
async def test_execute_with_valid_text_empty_result(mock_auto_tokenizer, mock_auto_model_for_token_classification, mock_transformers_pipeline_empty):
    tool = BiomedicalNERTool(model_name=settings.BIOMEDICAL_NER_MODEL_NAME)
    text = "This is a general sentence with no medical terms."
    result = await tool.execute(text=text) # Await the async method

    tool.pipe.assert_called_once_with(text)
    assert "entities" in result
    assert result["entities"] == []
    assert "error" not in result

@pytest.mark.asyncio
async def test_execute_with_empty_text(ner_tool_instance):
    result = await ner_tool_instance.execute(text="") # Await the async method
    assert "error" in result
    assert "non-empty string" in result["error"]
    assert "entities" not in result

@pytest.mark.asyncio
async def test_execute_with_non_string_input(ner_tool_instance):
    result = await ner_tool_instance.execute(text=123) # Await the async method
    assert "error" in result
    assert "non-empty string" in result["error"]
    assert "entities" not in result

@pytest.mark.asyncio
async def test_execute_pipeline_exception_handling(mock_auto_tokenizer, mock_auto_model_for_token_classification, mock_transformers_pipeline_exception):
    tool = BiomedicalNERTool(model_name=settings.BIOMEDICAL_NER_MODEL_NAME)
    text = "Some text that would cause an error"
    result = await tool.execute(text=text) # Await the async method
    
    tool.pipe.assert_called_once_with(text)
    assert "error" in result
    assert "Mock pipeline error" in result["error"]
    assert "entities" not in result

def test_tool_initialization_exception(mock_auto_tokenizer, mock_auto_model_for_token_classification):
    mock_auto_tokenizer.side_effect = Exception("Tokenizer load error")
    with pytest.raises(Exception, match="Tokenizer load error"):
        BiomedicalNERTool(model_name=settings.BIOMEDICAL_NER_MODEL_NAME)
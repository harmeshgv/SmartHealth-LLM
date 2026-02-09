# tests/backend/tools/test_google_search_tool.py
import pytest
from unittest.mock import patch, MagicMock
import logging
import pytest_asyncio

from app.tools.internet.google_search_tool import GoogleSearchTool
from app.config import settings

@pytest.fixture(autouse=True)
def cap_log(caplog):
    caplog.set_level(logging.WARNING)

@pytest.fixture
def mock_serper_wrapper():
    with patch('app.tools.internet.google_search_tool.GoogleSerperAPIWrapper') as MockClass:
        mock_instance = MockClass.return_value
        mock_instance.run.return_value = "Mocked search result for 'test query'"
        yield mock_instance

@pytest.fixture
def mock_settings_with_api_key(monkeypatch):
    monkeypatch.setattr(settings, 'SERPER_API_KEY', 'dummy_serper_api_key')

@pytest.fixture
def search_tool_instance(mock_settings_with_api_key, mock_serper_wrapper):
    return GoogleSearchTool()


def test_tool_initialization(search_tool_instance):
    assert search_tool_instance.name == "google_search"
    assert "Searches Google via Serper API" in search_tool_instance.description
    assert hasattr(search_tool_instance, 'search')

def test_tool_initialization_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, 'SERPER_API_KEY', None)
    with pytest.raises(ValueError, match="SERPER_API_KEY is missing in environment variables."):
        GoogleSearchTool()

@pytest.mark.asyncio
async def test_run_with_valid_query(search_tool_instance):
    query = "test query"
    result = await search_tool_instance.run(query=query) # Await the async method
    
    search_tool_instance.search.run.assert_called_once_with(query)
    
    assert "result" in result
    assert result["result"] == "Mocked search result for 'test query'"
    assert "error" not in result

@pytest.mark.asyncio
async def test_run_with_empty_query(search_tool_instance):
    result = await search_tool_instance.run(query="") # Await the async method
    assert "error" in result
    assert "non-empty string" in result["error"]
    assert "result" not in result

@pytest.mark.asyncio
async def test_run_with_non_string_query(search_tool_instance):
    result = await search_tool_instance.run(query={"not": "a string"}) # Await the async method
    assert "error" in result
    assert "non-empty string" in result["error"]
    assert "result" not in result

@pytest.mark.asyncio
async def test_run_api_error_handling(search_tool_instance):
    error_message = "Mock API Error"
    search_tool_instance.search.run.side_effect = Exception(error_message)
    
    query = "a query that will fail"
    result = await search_tool_instance.run(query=query) # Await the async method
    
    assert "error" in result
    assert error_message in result["error"]
    assert "result" not in result

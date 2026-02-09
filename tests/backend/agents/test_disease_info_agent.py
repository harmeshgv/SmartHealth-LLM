import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

from app.agents.roles.disease_info_agent import DiseaseInfoAgent
from app.core.agent_context import AgentContext

@pytest.fixture
def mock_agent_context():
    """Provides a mocked AgentContext."""
    context = MagicMock(spec=AgentContext)
    context.session_id = "test_session"
    return context

@pytest.fixture
def mock_tools():
    """Mocks the tools used by the DiseaseInfoAgent."""
    with (patch('app.agents.roles.disease_info_agent.BiomedicalNERTool') as MockNER, 
          patch('app.agents.roles.disease_info_agent.DiseaseInfoRetrieverTool') as MockDB, 
          patch('app.agents.roles.disease_info_agent.GoogleSearchTool') as MockGoogle):
        
        mock_ner_instance = MockNER.return_value
        mock_ner_instance.run = AsyncMock(return_value={"diseases": ["diabetes"]})

        mock_db_instance = MockDB.return_value
        mock_db_instance.run = AsyncMock(return_value={"info": "Diabetes is a chronic condition..."})

        mock_google_instance = MockGoogle.return_value
        mock_google_instance.run = AsyncMock(return_value={"info": "Web search result for Diabetes"})

        yield mock_ner_instance, mock_db_instance, mock_google_instance

@pytest.fixture
def disease_info_agent(mock_agent_context, mock_tools):
    """Fixture to create a DiseaseInfoAgent instance with mocked dependencies."""
    with patch('app.agents.base.base_agent.BaseAgent.__init__', return_value=None):
        agent = DiseaseInfoAgent(mock_agent_context)
        agent.context = mock_agent_context
        agent.session_id = "test_session"
        agent.ner_tool, agent.db_tool, agent.google = mock_tools
        agent.llm_call = AsyncMock(return_value=json.dumps({"disease": "diabetes"}))
        agent.save_memory = AsyncMock()
        yield agent

@pytest.mark.asyncio
@patch('app.agents.roles.disease_info_agent.load_prompt', return_value="Prompt: {{INPUT}} {{NER_CONTEXT}}")
async def test_run_graph_path_db_success(mock_load_prompt, disease_info_agent, mock_tools):
    """
    Tests the graph path where the local DB search is successful.
    """
    mock_ner, mock_db, mock_google = mock_tools
    user_message = "Tell me about diabetes"
    run_id = "run-graph-db-success"

    result = await disease_info_agent.run(user_message=user_message, run_id=run_id)

    # Assertions
    mock_ner.run.assert_awaited_once_with(user_message, run_id=run_id)
    disease_info_agent.llm_call.assert_awaited_once()
    mock_db.run.assert_awaited_once_with("diabetes", run_id=run_id)
    mock_google.run.assert_not_awaited() # Should not be called
    disease_info_agent.save_memory.assert_awaited_once()

    assert result["disease"] == "diabetes"
    assert result["info"] == {"info": "Diabetes is a chronic condition..."}

@pytest.mark.asyncio
@patch('app.agents.roles.disease_info_agent.load_prompt', return_value="Prompt: {{INPUT}} {{NER_CONTEXT}}")
async def test_run_graph_path_db_fail_google_success(mock_load_prompt, disease_info_agent, mock_tools):
    """
    Tests the graph path where the local DB fails and it falls back to Google search.
    """
    mock_ner, mock_db, mock_google = mock_tools
    # Simulate DB tool failing to find info
    mock_db.run.return_value = None
    user_message = "Tell me about a rare disease"
    run_id = "run-graph-db-fail"
    
    # Mock LLM to extract the disease
    disease_info_agent.llm_call.return_value = json.dumps({"disease": "rare_disease"})
    mock_ner.run.return_value = {"diseases": ["rare_disease"]}

    result = await disease_info_agent.run(user_message=user_message, run_id=run_id)

    # Assertions
    mock_ner.run.assert_awaited_once_with(user_message, run_id=run_id)
    mock_db.run.assert_awaited_once_with("rare_disease", run_id=run_id)
    mock_google.run.assert_awaited_once_with("rare_disease", run_id=run_id)
    disease_info_agent.save_memory.assert_awaited_once()

    assert result["disease"] == "rare_disease"
    assert result["info"] == {"info": "Web search result for Diabetes"} # Using the mock google's return value

@pytest.mark.asyncio
async def test_run_direct_path_db_success(disease_info_agent, mock_tools):
    """
    Tests the direct path where a disease is passed directly and found in the DB.
    """
    mock_ner, mock_db, mock_google = mock_tools
    run_id = "run-direct-db-success"
    
    result = await disease_info_agent.run(
        user_message="", 
        run_id=run_id, 
        disease_matched={"disease": "diabetes"}
    )

    # Assertions
    mock_db.run.assert_awaited_once_with("diabetes", run_id=run_id)
    mock_google.run.assert_not_awaited()
    mock_ner.run.assert_not_awaited() # Should not enter the graph
    
    assert result["disease"] == "diabetes"
    assert result["info"] == {"info": "Diabetes is a chronic condition..."}

@pytest.mark.asyncio
async def test_run_direct_path_db_fail(disease_info_agent, mock_tools):
    """
    Tests the direct path where a disease is passed, not found in DB, and falls back to Google.
    """
    mock_ner, mock_db, mock_google = mock_tools
    mock_db.run.return_value = None # Simulate DB failure
    run_id = "run-direct-db-fail"

    result = await disease_info_agent.run(
        user_message="", 
        run_id=run_id, 
        disease_matched={"disease": "rare_disease"}
    )
    
    # Assertions
    mock_db.run.assert_awaited_once_with("rare_disease", run_id=run_id)
    mock_google.run.assert_awaited_once_with("rare_disease", run_id=run_id)
    mock_ner.run.assert_not_awaited()
    
    assert result["disease"] == "rare_disease"
    assert result["info"] == {"info": "Web search result for Diabetes"}

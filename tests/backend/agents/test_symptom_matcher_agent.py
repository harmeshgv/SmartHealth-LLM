import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from app.agents.roles.symptom_matcher_agent import SymptomMatcherAgent
from app.core.agent_context import AgentContext

@pytest.fixture
def mock_agent_context():
    """Provides a mocked AgentContext."""
    context = MagicMock(spec=AgentContext)
    context.session_id = "test_session"
    return context

@pytest.fixture
def mock_tools():
    """Mocks the tools used by the SymptomMatcherAgent."""
    with (patch('app.agents.roles.symptom_matcher_agent.BiomedicalNERTool') as MockNER, 
          patch('app.agents.roles.symptom_matcher_agent.SymptomDiseaseMatcherTool') as MockMatcher):
        
        mock_ner_instance = MockNER.return_value
        mock_ner_instance.run = AsyncMock(return_value={"symptoms": ["fever", "cough"]})

        mock_matcher_instance = MockMatcher.return_value
        mock_matcher_instance.run = AsyncMock(return_value={
            "matched_diseases": [
                {"disease": "Common Cold", "score": 0.9},
                {"disease": "Flu", "score": 0.8}
            ]
        })

        yield mock_ner_instance, mock_matcher_instance

@pytest.fixture
def symptom_matcher_agent(mock_agent_context, mock_tools):
    """Fixture to create a SymptomMatcherAgent instance with mocked dependencies."""
    with patch('app.agents.base.base_agent.BaseAgent.__init__', return_value=None):
        agent = SymptomMatcherAgent(mock_agent_context)
        agent.context = mock_agent_context
        agent.session_id = "test_session"
        agent.ner_tool, agent.symp_match_tool = mock_tools
        agent.llm_call = AsyncMock(return_value=json.dumps({"symptoms": ["fever", "cough"]}))
        agent.save_memory = AsyncMock()
        yield agent

@pytest.mark.asyncio
@patch('app.agents.roles.symptom_matcher_agent.load_prompt', return_value="Extract symptoms: {{INPUT}} {{NER_CONTEXT}}")
async def test_symptom_matcher_agent_run_success(mock_load_prompt, symptom_matcher_agent, mock_tools):
    """
    Tests the successful run of the SymptomMatcherAgent.
    """
    mock_ner, mock_matcher = mock_tools
    user_message = "I have a fever and a cough."
    run_id = "run-123"

    result = await symptom_matcher_agent.run(user_message=user_message, run_id=run_id)

    # Assertions
    mock_ner.run.assert_awaited_once_with(user_message, run_id=run_id)
    symptom_matcher_agent.llm_call.assert_awaited_once()
    mock_load_prompt.assert_called_once_with("symptom_matcher_prompt.txt")
    
    # Check that the matcher tool was called with the symptoms from the parse_node
    mock_matcher.run.assert_awaited_once_with(["fever", "cough"], run_id=run_id)
    
    symptom_matcher_agent.save_memory.assert_awaited_once()
    
    assert result["symptoms"] == ["fever", "cough"]
    assert result["disease_matched"] == "Common Cold" # Should be the top match

@pytest.mark.asyncio
async def test_matcher_node_logic(symptom_matcher_agent, mock_tools):
    """
    Directly tests the logic inside the matcher_node for different tool outputs.
    """
    _, mock_matcher = mock_tools
    
    # Case 1: Standard successful output
    state = {"symptoms": ["fever"], "run_id": "test-run"}
    result_state = await symptom_matcher_agent.matcher_node(state)
    assert result_state["disease_matched"] == "Common Cold"

    # Case 2: Tool returns an empty list
    mock_matcher.run.return_value = {"matched_diseases": []}
    state = {"symptoms": ["unknown symptom"], "run_id": "test-run"}
    result_state = await symptom_matcher_agent.matcher_node(state)
    assert result_state["disease_matched"] == "unknown"

    # Case 3: Tool returns a string (as a fallback)
    mock_matcher.run.return_value = "Flu"
    state = {"symptoms": ["fatigue"], "run_id": "test-run"}
    result_state = await symptom_matcher_agent.matcher_node(state)
    assert result_state["disease_matched"] == "Flu"

    # Case 4: Tool returns something unexpected
    mock_matcher.run.return_value = {"unexpected_key": "some_value"}
    state = {"symptoms": ["dizziness"], "run_id": "test-run"}
    result_state = await symptom_matcher_agent.matcher_node(state)
    assert result_state["disease_matched"] == "unknown"

@pytest.mark.asyncio
async def test_parse_node_fallback(symptom_matcher_agent):
    """
    Tests the parse_node's fallback when LLM output is not valid JSON.
    """
    state = {"llm_output": "not json", "run_id": "test-run"}
    result_state = await symptom_matcher_agent.parse_node(state)
    assert result_state["symptoms"] == ["None"]

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from app.agents.roles.decider_agent import DeciderAgent
from app.core.agent_context import AgentContext

@pytest.fixture
def mock_agent_context():
    """Provides a mocked AgentContext."""
    context = MagicMock(spec=AgentContext)
    context.session_id = "test_session"
    return context

@pytest.fixture
def decider_agent(mock_agent_context):
    """Fixture to create a DeciderAgent instance with mocked dependencies."""
    with patch('app.agents.base.base_agent.BaseAgent.__init__', return_value=None):
        agent = DeciderAgent(mock_agent_context)
        agent.context = mock_agent_context
        agent.session_id = "test_session"
        agent.llm_call = AsyncMock() # Will be configured in each test
        yield agent

@pytest.mark.asyncio
@patch('app.agents.roles.decider_agent.load_prompt', return_value="Decide: {{INPUT}}")
async def test_decider_agent_run_successful_parse(mock_load_prompt, decider_agent):
    """
    Tests the successful run of the DeciderAgent where the LLM returns valid JSON.
    """
    user_message = "What are the symptoms of diabetes?"
    run_id = "run-123"
    
    # Mock the LLM to return a valid JSON string
    llm_response = {
        "intent": "disease_information",
        "agents": ["symptom_matcher_agent", "disease_info_agent"]
    }
    decider_agent.llm_call.return_value = json.dumps(llm_response)

    # Execute the agent's run method
    result = await decider_agent.run(user_message=user_message, run_id=run_id)

    # Assertions
    mock_load_prompt.assert_called_once_with("decider_prompt.txt")
    decider_agent.llm_call.assert_awaited_once_with(f"Decide: {user_message}", run_id=run_id)
    
    assert result["intent"] == "disease_information"
    assert result["agents"] == ["symptom_matcher_agent", "disease_info_agent"]

@pytest.mark.asyncio
@patch('app.agents.roles.decider_agent.load_prompt', return_value="Decide: {{INPUT}}")
async def test_decider_agent_run_failed_parse(mock_load_prompt, decider_agent):
    """
    Tests the fallback mechanism of the DeciderAgent when the LLM returns invalid JSON.
    """
    user_message = "Just a casual chat."
    run_id = "run-456"

    # Mock the LLM to return a non-JSON string
    decider_agent.llm_call.return_value = "This is not JSON."

    # Execute the agent's run method
    result = await decider_agent.run(user_message=user_message, run_id=run_id)

    # Assertions
    decider_agent.llm_call.assert_awaited_once()
    
    # Check that it returned the fallback response
    assert result["intent"] == "fallback"
    assert result["agents"] == ["conversation_agent"]

@pytest.mark.asyncio
async def test_parse_node_valid_json(decider_agent):
    """Directly tests the parse_node with valid JSON."""
    valid_json_str = '{"intent": "test_intent", "agents": ["conversation_agent"]}'
    initial_state = {"llm_output": valid_json_str, "run_id": "test-run"}
    
    result_state = await decider_agent.parse_node(initial_state)
    
    assert result_state["intent"] == "test_intent"
    assert result_state["agents"] == ["conversation_agent"]

@pytest.mark.asyncio
async def test_parse_node_invalid_json(decider_agent):
    """Directly tests the parse_node with invalid JSON."""
    invalid_json_str = 'this is not valid json'
    initial_state = {"llm_output": invalid_json_str, "run_id": "test-run"}
    
    result_state = await decider_agent.parse_node(initial_state)
    
    assert result_state["intent"] == "fallback"
    assert result_state["agents"] == ["conversation_agent"]

@pytest.mark.asyncio
async def test_parse_node_missing_keys(decider_agent):
    """Tests the parse_node when the JSON is missing expected keys."""
    json_str = '{"some_other_key": "some_value"}'
    initial_state = {"llm_output": json_str, "run_id": "test-run"}
    
    result_state = await decider_agent.parse_node(initial_state)
    
    assert result_state["intent"] == "conversation"
    assert result_state["agents"] == ["conversation_agent"]

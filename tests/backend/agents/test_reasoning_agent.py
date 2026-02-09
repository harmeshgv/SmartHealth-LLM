import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from app.agents.roles.reasoning_agent import ReasoningAgent
from app.core.agent_context import AgentContext

@pytest.fixture
def mock_agent_context():
    """Provides a mocked AgentContext."""
    context = MagicMock(spec=AgentContext)
    context.session_id = "test_session"
    return context

@pytest.fixture
def reasoning_agent(mock_agent_context):
    """Fixture to create a ReasoningAgent instance with mocked dependencies."""
    with patch('app.agents.base.base_agent.BaseAgent.__init__', return_value=None):
        agent = ReasoningAgent(mock_agent_context)
        agent.context = mock_agent_context
        agent.session_id = "test_session"
        agent.llm_call = AsyncMock(return_value="This is the final synthesized response.")
        yield agent

@pytest.mark.asyncio
@patch('app.agents.roles.reasoning_agent.load_prompt', return_value="User: {{USER_MESSAGE}}, Symptoms: {{SYMPTOMS}}, Disease: {{DISEASE}}, Info: {{DISEASE_INFO}}")
async def test_reasoning_agent_run_with_all_info(mock_load_prompt, reasoning_agent):
    """
    Tests the ReasoningAgent's run method with a full set of context from previous agents.
    """
    user_message = "I have a fever and a cough, what could it be?"
    run_id = "run-123"
    kwargs = {
        "symptoms": ["fever", "cough"],
        "disease_matched": "Common Cold",
        "info": {"overview": "The common cold is a viral infection..."}
    }

    result = await reasoning_agent.run(user_message=user_message, run_id=run_id, **kwargs)

    # Assertions
    mock_load_prompt.assert_called_once_with("reasoning_prompt.txt")

    # Check that the prompt was formatted correctly
    expected_prompt = (
        f"User: {user_message}, "
        f"Symptoms: {json.dumps(kwargs['symptoms'])}, "
        f"Disease: {kwargs['disease_matched']}, "
        f"Info: {json.dumps(kwargs['info'])}"
    )
    reasoning_agent.llm_call.assert_awaited_once_with(expected_prompt, run_id=run_id)

    # Check the final output
    assert result["final_output"] == "This is the final synthesized response."

@pytest.mark.asyncio
@patch('app.agents.roles.reasoning_agent.load_prompt', return_value="User: {{USER_MESSAGE}}, Symptoms: {{SYMPTOMS}}, Disease: {{DISEASE}}, Info: {{DISEASE_INFO}}")
async def test_reasoning_agent_run_with_missing_info(mock_load_prompt, reasoning_agent):
    """
    Tests the ReasoningAgent's run method when some context is missing.
    """
    user_message = "What is a migraine?"
    run_id = "run-456"
    kwargs = {
        "disease_matched": "Migraine",
        # Missing symptoms and info
    }

    await reasoning_agent.run(user_message=user_message, run_id=run_id, **kwargs)

    # Assert that the prompt is still formatted correctly with default values
    expected_prompt = (
        f"User: {user_message}, "
        f"Symptoms: {json.dumps([])}, "
        f"Disease: {kwargs['disease_matched']}, "
        f"Info: {json.dumps({})}"
    )
    reasoning_agent.llm_call.assert_awaited_once_with(expected_prompt, run_id=run_id)
    
@pytest.mark.asyncio
@patch('app.agents.roles.reasoning_agent.load_prompt', return_value="User: {{USER_MESSAGE}}, Symptoms: {{SYMPTOMS}}, Disease: {{DISEASE}}, Info: {{DISEASE_INFO}}")
async def test_reasoning_agent_run_with_no_extra_kwargs(mock_load_prompt, reasoning_agent):
    """
    Tests the ReasoningAgent's run method with no extra kwargs provided.
    """
    user_message = "Hello"
    run_id = "run-789"
    
    await reasoning_agent.run(user_message=user_message, run_id=run_id)

    # Assert that the prompt uses default values for all context fields
    expected_prompt = (
        f"User: {user_message}, "
        f"Symptoms: {json.dumps([])}, "
        f"Disease: unknown, "
        f"Info: {json.dumps({})}"
    )
    reasoning_agent.llm_call.assert_awaited_once_with(expected_prompt, run_id=run_id)

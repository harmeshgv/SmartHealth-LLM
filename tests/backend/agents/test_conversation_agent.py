import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.agents.roles.conversation_agent import ConversationAgent
from app.core.agent_context import AgentContext

@pytest.fixture
def mock_agent_context():
    """Provides a mocked AgentContext."""
    context = MagicMock(spec=AgentContext)
    context.session_id = "test_session"
    context.tools = {}
    return context

@pytest.fixture
def conversation_agent(mock_agent_context):
    """Fixture to create a ConversationAgent instance with mocked dependencies."""
    # Mock the base agent's __init__ to avoid real LLM factory calls
    with patch('app.agents.base.base_agent.BaseAgent.__init__', return_value=None) as mock_base_init:
        agent = ConversationAgent(mock_agent_context)
        # Manually set the mocked context and other necessary attributes
        agent.context = mock_agent_context
        agent.session_id = "test_session"
        agent.llm_call = AsyncMock(return_value="Mocked LLM response")
        agent.recall_memory = AsyncMock(return_value=[])
        agent.save_memory = AsyncMock()
        mock_base_init.assert_called_once_with(mock_agent_context)
        return agent

@pytest.mark.asyncio
@patch('app.agents.roles.conversation_agent.load_prompt', return_value="Prompt: {{PAST}} {{INPUT}}")
async def test_conversation_agent_run(mock_load_prompt, conversation_agent):
    """
    Tests the full run method of the ConversationAgent.
    """
    user_message = "Hello, agent!"
    run_id = "run-123"

    # Execute the agent's run method
    result = await conversation_agent.run(user_message=user_message, run_id=run_id)

    # 1. Assert recall_memory was called
    conversation_agent.recall_memory.assert_awaited_once_with(run_id=run_id, n=5)

    # 2. Assert load_prompt was called
    mock_load_prompt.assert_called_once_with("conversation_prompt.txt")

    # 3. Assert llm_call was made with the correct prompt
    expected_prompt = "Prompt:  " + user_message # Empty past_text
    conversation_agent.llm_call.assert_awaited_once_with(expected_prompt, run_id=run_id)

    # 4. Assert save_memory was called
    conversation_agent.save_memory.assert_awaited_once_with(
        user_message=user_message,
        llm_output="Mocked LLM response",
        run_id=run_id
    )

    # 5. Assert the final result is correct
    assert result == {"llm_output": "Mocked LLM response"}

@pytest.mark.asyncio
@patch('app.agents.roles.conversation_agent.load_prompt', return_value="Prompt: {{PAST}} {{INPUT}}")
async def test_conversation_agent_with_memory(mock_load_prompt, conversation_agent):
    """
    Tests that the agent correctly formats and uses past conversation history.
    """
    user_message = "What was the last thing I said?"
    run_id = "run-456"
    
    # Simulate recalling a past conversation
    past_memory = [
        {"user_message": "Hello, agent!", "agent_output": "Hi there!"}
    ]
    conversation_agent.recall_memory.return_value = past_memory

    # Execute the agent's run method
    await conversation_agent.run(user_message=user_message, run_id=run_id)

    # Assert that the prompt includes the formatted past messages
    past_text = "User: Hello, agent!\nAssistant: Hi there!"
    expected_prompt = f"Prompt: {past_text} {user_message}"
    conversation_agent.llm_call.assert_awaited_once_with(expected_prompt, run_id=run_id)

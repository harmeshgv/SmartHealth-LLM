import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from app.core.agent_orchestrator import AgentOrchetrator
from app.core.agent_context import AgentContext

@pytest.fixture
def mock_agent_context():
    """Provides a mocked AgentContext."""
    return MagicMock(spec=AgentContext)

@pytest.fixture
def mock_agent_factory():
    """Mocks the AgentFactory to return controllable mock agents."""
    with patch('app.core.agent_orchestrator.AgentFactory') as MockFactory:
        factory_instance = MockFactory.return_value
        
        # Create mock agents
        mock_decider = MagicMock()
        mock_decider.run = AsyncMock(return_value={"agents": ["symptom_matcher_agent", "disease_info_agent"]})

        mock_symptom_matcher = MagicMock()
        mock_symptom_matcher.run = AsyncMock(return_value={"symptoms": ["fever"], "disease_matched": "Flu"})

        mock_disease_info = MagicMock()
        mock_disease_info.run = AsyncMock(return_value={"info": "Flu is a viral infection."})

        mock_reasoner = MagicMock()
        mock_reasoner.run = AsyncMock(return_value={"final_output": "Based on your symptoms, you might have the Flu."})

        # Configure the factory to return the correct mock for each agent name
        agent_map = {
            "decider_agent": mock_decider,
            "symptom_matcher_agent": mock_symptom_matcher,
            "disease_info_agent": mock_disease_info,
            "reasoning_agent": mock_reasoner
        }
        factory_instance.create.side_effect = lambda agent_name: agent_map[agent_name]
        
        yield factory_instance, agent_map

@pytest.mark.asyncio
async def test_agent_orchestrator_run(mock_agent_context, mock_agent_factory):
    """
    Tests the main run method of the AgentOrchestrator to ensure it
    correctly sequences agents based on the decider's plan.
    """
    factory_instance, agent_map = mock_agent_factory
    
    orchestrator = AgentOrchetrator(mock_agent_context)
    user_message = "I have a fever and feel sick."
    
    # Run the orchestrator
    result = await orchestrator.run(user_message)

    # 1. Assert the decider agent was called correctly
    decider = agent_map["decider_agent"]
    decider.run.assert_awaited_once_with(user_message, run_id=ANY)

    # 2. Assert the agents in the plan were called in order
    symptom_matcher = agent_map["symptom_matcher_agent"]
    symptom_matcher.run.assert_awaited_once()
    
    disease_info = agent_map["disease_info_agent"]
    disease_info.run.assert_awaited_once()

    # Check that the state was passed correctly between agents
    # symptom_matcher is called with the initial state
    symptom_matcher_call_args = symptom_matcher.run.call_args[1]
    assert symptom_matcher_call_args['user_message'] == user_message
    
    # disease_info is called with the state updated by symptom_matcher
    disease_info_call_args = disease_info.run.call_args[1]
    assert disease_info_call_args['user_message'] == user_message
    assert disease_info_call_args['symptoms'] == ["fever"]
    assert disease_info_call_args['disease_matched'] == "Flu"
    
    # 3. Assert the reasoning agent was called with the final state
    reasoner = agent_map["reasoning_agent"]
    reasoner.run.assert_awaited_once()
    reasoner_call_args = reasoner.run.call_args[1]
    assert reasoner_call_args['user_message'] == user_message
    assert reasoner_call_args['symptoms'] == ["fever"]
    assert reasoner_call_args['disease_matched'] == "Flu"
    assert reasoner_call_args['info'] == "Flu is a viral infection."

    # 4. Assert the final output is correct
    assert result["final_output"] == "Based on your symptoms, you might have the Flu."
    assert "run_id" in result

import pytest
from fastapi.testclient import TestClient # Import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from app.main import app
from app.core.agent_orchestrator import AgentOrchetrator
from app.core.agent_context import AgentContext

# Create a synchronous TestClient instance
client = TestClient(app)

@pytest.mark.asyncio
async def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SmartHealth Backend Running"}

@pytest.mark.asyncio
async def test_health_endpoint():
    response = client.get("/health/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
@patch('app.api.v1.chat.AgentOrchetrator')
async def test_send_message_success(MockAgentOrchetrator):
    mock_orchestrator_instance = MockAgentOrchetrator.return_value
    mock_orchestrator_instance.run = AsyncMock(return_value={"final_output": "Mocked AI Response", "run_id": "mock_run_id"})

    response = client.post("/chat/send", json={"message": "Hello", "session_id": "test_session"})
    
    assert response.status_code == 200
    assert response.json() == {"reply": {"final_output": "Mocked AI Response", "run_id": "mock_run_id"}, "session_id": "test_session"}
    MockAgentOrchetrator.assert_called_once()
    mock_orchestrator_instance.run.assert_awaited_once_with("Hello")

@pytest.mark.asyncio
async def test_send_message_empty_message():
    response = client.post("/chat/send", json={"message": "", "session_id": "test_session"})
    
    assert response.status_code == 400
    assert response.json() == {"detail": "Message cannot be empty"}

@pytest.mark.asyncio
@patch('app.api.v1.chat.AgentContext')
async def test_clear_chat(MockAgentContext):
    mock_context_instance = MockAgentContext.return_value
    mock_context_instance.session_id = "test_session" # Set the session_id attribute
    mock_context_instance.long_memory.clear = AsyncMock()

    response = client.post("/chat/clear", json={"session_id": "test_session"})
    
    assert response.status_code == 200
    assert response.json() == {"message": "Session Cleared"}
    MockAgentContext.assert_called_once_with(session_id="test_session")
    mock_context_instance.long_memory.clear.assert_awaited_once_with("test_session")

@pytest.mark.asyncio
@patch('app.api.v1.chat.AgentContext')
async def test_chat_history(MockAgentContext):
    mock_history = [
        {"user_message": "Hi", "agent_output": "Hello"},
        {"user_message": "How are you?", "agent_output": "I'm good!"}
    ]
    mock_context_instance = MockAgentContext.return_value
    mock_context_instance.session_id = "test_session" # Set the session_id attribute
    mock_context_instance.long_memory.get = AsyncMock(return_value=mock_history)

    response = client.post("/chat/history", json={"session_id": "test_session"})
    
    assert response.status_code == 200
    assert response.json() == {"history": mock_history}
    MockAgentContext.assert_called_once_with(session_id="test_session")
    mock_context_instance.long_memory.get.assert_awaited_once_with("test_session")

@pytest.mark.asyncio
@patch('app.api.v1.debug.AgentOrchetrator.run', new_callable=AsyncMock) # Patch the run method directly
@patch('app.api.v1.debug.io.StringIO') # This is the second patch
async def test_debug_chat_send(MockStringIO, mock_run_method): # Corrected order
    mock_run_method.return_value = {"final_output": "Debug AI Response", "run_id": "debug_run_id"}

    # Configure the mock StringIO to return a dummy log string
    mock_string_io_instance = MockStringIO.return_value
    mock_string_io_instance.getvalue.return_value = "Mocked debug log output."

    response = client.post("/debug/debug_chat_send", json={"message": "Debug me!", "session_id": "debug_session"})
    
    assert response.status_code == 200
    assert response.json()["reply"] == {"final_output": "Debug AI Response", "run_id": "debug_run_id"}
    assert response.json()["session_id"] == "debug_session"
    assert "debug_logs" in response.json()
    assert isinstance(response.json()["debug_logs"], str)
    assert response.json()["debug_logs"] == "Mocked debug log output." # Assert content
    mock_run_method.assert_awaited_once_with("Debug me!")


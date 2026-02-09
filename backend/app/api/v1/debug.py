import io
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.core.agent_context import AgentContext
from app.core.agent_orchestrator import AgentOrchetrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["debug"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    image: Optional[str] = None  # base64 string


@router.post("/debug_chat_send")
async def debug_chat_send(data: ChatRequest):
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Store original level and add the debug handler
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG) # Temporarily set root logger to DEBUG to capture all messages
    root_logger.addHandler(handler)

    try:
        context = AgentContext(session_id=data.session_id)
        orchetrator = AgentOrchetrator(context)
        reply = await orchetrator.run(data.message)
    finally:
        # Clean up: remove handler and restore original level
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)
    
    captured_logs = log_stream.getvalue()
    return {"reply": reply, "session_id": context.session_id, "debug_logs": captured_logs}

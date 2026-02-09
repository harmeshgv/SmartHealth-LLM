from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.core.agent_orchestrator import AgentOrchetrator
from app.core.agent_context import AgentContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    image: Optional[str] = None  # base64 string


class SessionRequest(BaseModel):
    session_id: str


@router.post("/send")
async def send_message(data: ChatRequest):

    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    context = AgentContext(session_id=data.session_id)

    orchestrator = AgentOrchetrator(context)

    reply = await orchestrator.run(data.message)

    return {"reply": reply, "session_id": context.session_id}


@router.post("/clear")
async def clear_chat(data: SessionRequest):
    context = AgentContext(session_id=data.session_id)
    await context.long_memory.clear(context.session_id)
    return {"message": "Session Cleared"}


@router.post("/history")
async def chat_history(data: SessionRequest):
    context = AgentContext(session_id=data.session_id)
    all_history = await context.long_memory.get(context.session_id)
    return {"history": all_history}

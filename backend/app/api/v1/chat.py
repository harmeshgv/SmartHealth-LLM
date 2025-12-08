from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.dependencies import get_current_user
from app.core.agent_orchestrator import AgentOrchetrator
from app.core.agent_context import AgentContext

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str
    image: Optional[str] = None   # base64 string

@router.post("/send")
async def send_message(data: ChatRequest, user=Depends(get_current_user)):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Create context with user
    context = AgentContext(user=user)

    orchestrator = AgentOrchetrator(context)

    reply = await orchestrator.run(data.message)

    return {"reply": reply}



@router.post("/clear")
async def clear_chat(user=Depends(get_current_user)):
    return {"message": "Session Cleared"}


from app.core.agent_context import AgentContext

@router.get("/history")
async def chat_history(user=Depends(get_current_user)):
    context = AgentContext(user=user)

    all_history = await context.long_memory.get(context.session_id)

    return {"history": all_history}


@router.get("/debug")
async def debug_chat(user=Depends(get_current_user)):
    return {"debug": "hi"}

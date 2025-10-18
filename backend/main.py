from fastapi import FastAPI
from pydantic import BaseModel
from utils.llm import set_llm
from dotenv import load_dotenv
import os
import uuid
from agent_orchestrator import AgentOrchestration


app = FastAPI()

user_orchestrations = {}


class SetupLLMRequest(BaseModel):
    api_key: str
    provider: str
    model: str


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/setup_llm")
def setup_llm(request: SetupLLMRequest):
    llm_instance = set_llm(
        api_key=request.api_key, provider=request.provider, model=request.model
    )
    user_id = str(uuid.uuid4())
    orchestrator = AgentOrchestration(llm_instance)
    user_orchestrations[user_id] = orchestrator

    return {"status": "success", "message": f"LLM set up for user {user_id}"}


@app.post("/ask")
def ask(request: ChatRequest):
    # Replace with your actual LLM/agent logic
    if request.user_id not in user_orchestrations:
        return {"error": "LLM not set up for this user"}

    orchestrator_instance = user_orchestrations[request.user_id]

    response = orchestrator_instance.main(
        request.message
    )  # depends on your LLM wrapper
    return {"answer": response}

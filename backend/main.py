from fastapi import FastAPI
from pydantic import BaseModel
from backend.agents.decider_agent import DECIDERAGENT

app = FastAPI()
DC = DECIDERAGENT()

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask(query: Query):
    # Replace with your actual LLM/agent logic
    answer = DC.main(query.query)
    return {"answer": answer}

from fastapi import FastAPI
from pydantic import BaseModel
from agents.decider_agent import DECIDERAGENT
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
DC = DECIDERAGENT()
origins = [
    "http://localhost:3000",  # React dev server
    # Add your production frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allows your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str

@app.get("/")
def health():
    return {"status":"ok"}

@app.post("/ask")
def ask(query: Query):
    # Replace with your actual LLM/agent logic
    answer = DC.main(query.query)
    return {"answer": answer}

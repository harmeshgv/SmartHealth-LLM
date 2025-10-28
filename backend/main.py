# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from agent_orchestrator import AgentOrchestration
from utils.llm import set_llm
import logging
from typing import Optional
import base64
from io import BytesIO
from PIL import Image

# initialize logging
logging.basicConfig(level=logging.CRITICAL, filename="app.log", filemode="a")

logger = logging.getLogger("main")

logger.info("Application started")

# Load environment variables
load_dotenv()
logger.info("Application started")

# Initialize LLM instance
llm_instance = set_llm(
    os.getenv("TEST_API_KEY"),
    os.getenv("TEST_API_BASE"),
    os.getenv("TEST_MODEL"),
)

# Initialize FastAPI app
app = FastAPI(title="Smart Health LLM API")

# Add CORS middleware (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://harmeshgv.github.io/SmartHealth-LLM/",
    ],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store orchestrators per user
user_orchestrations = {}


# Request model
class ChatRequest(BaseModel):
    user_id: str
    message: str
    image: Optional[str] = None  # base64 string


# Health check endpoint
@app.get("/")
def health():
    return {"status": "ok"}


# Chat endpoint
@app.post("/ask")
def ask(request: ChatRequest):
    if request.user_id not in user_orchestrations:
        try:
            user_orchestrations[request.user_id] = AgentOrchestration(llm_instance)
        except Exception as e:
            return {"error": f"Failed to initialize AI agent: {str(e)}"}

    orchestrator_instance = user_orchestrations[request.user_id]

    # Convert base64 image to PIL Image if present
    image = None
    if request.image:
        try:
            header, base64_data = request.image.split(",", 1)
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}

    # Invoke orchestrator
    try:
        response = orchestrator_instance.invoke(user_query=request.message, image=image)
        return {"answer": response}
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}


# Run app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

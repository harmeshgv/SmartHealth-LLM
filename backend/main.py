# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional
import base64
from io import BytesIO
from PIL import Image
import uvicorn

from .config import settings
from .utils.logging import setup_logging # Import the setup function
from .agent_orchestrator import AgentOrchestration
from .utils.llm import set_llm
from .tool_registry import tool_registry

# Set up logging before anything else
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)

logger = logging.getLogger(__name__)

logger.info("Application starting...")

# Initialize LLM instance (now using settings)
llm_instance = set_llm(
    settings.TEST_API_KEY,
    settings.TEST_API_BASE,
    settings.TEST_MODEL,
)
logger.info("LLM initialized.")

# Initialize FastAPI app
app = FastAPI(title="Smart Health LLM API")

# Add CORS middleware (using settings.CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import redis
import dill # Using dill for better serialization of complex objects

# Initialize Redis connection
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=False  # Store bytes to handle pickled objects
    )
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Could not connect to Redis: {e}", exc_info=True)
    redis_client = None

# Request model
class ChatRequest(BaseModel):
    user_id: str
    message: str
    image: Optional[str] = None  # base64 string


# Health check endpoint
@app.get("/")
def health():
    if redis_client and redis_client.ping():
        return {"status": "ok", "redis_status": "ok"}
    return {"status": "ok", "redis_status": "error"}


# Chat endpoint
@app.post("/ask")
async def ask(request: ChatRequest): # Made async
    if not redis_client:
        return {"error": "Redis is not available. Please check the server configuration."}

    session_key = f"session:{request.user_id}"
    
    # Always create a new orchestrator instance to ensure it's stateless
    orchestrator_instance = AgentOrchestration(llm_instance, tool_registry)
    logger.info(f"Initialized new AI agent for user: {request.user_id}")

    try:
        # Try to retrieve existing session memory from Redis
        serialized_memory = redis_client.get(session_key)
        
        if serialized_memory:
            retrieved_memory = dill.loads(serialized_memory)
            orchestrator_instance.set_memory(retrieved_memory)
            logger.info(f"Retrieved session memory for user: {request.user_id}")
            
    except Exception as e:
        logger.error(f"Failed to retrieve or deserialize session for user {request.user_id}: {str(e)}", exc_info=True)
        # Continue with a fresh session if retrieval fails
    
    # Convert base64 image to PIL Image if present
    image = None
    if request.image:
        try:
            # Split off the header if present (e.g., "data:image/jpeg;base64,")
            if "," in request.image:
                header, base64_data = request.image.split(",", 1)
            else:
                base64_data = request.image

            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            logger.info("Processed image from request.")
        except Exception as e:
            logger.error(f"Failed to process image for user {request.user_id}: {str(e)}", exc_info=True)
            return {"error": f"Failed to process image: {str(e)}"}

    # Invoke orchestrator
    try:
        # Invoke the orchestrator (now async)
        response = await orchestrator_instance.invoke(user_query=request.message, image=image)
        logger.info(f"Response Generated Successfully for user: {request.user_id}")

        # Get the updated memory and save it back to Redis
        updated_memory = orchestrator_instance.get_memory()
        serialized_memory = dill.dumps(updated_memory)
        redis_client.setex(session_key, settings.SESSION_EXPIRE_SECONDS, serialized_memory)
        
        return {"answer": response}
    except Exception as e:
        logger.error(f"Internal server error for user {request.user_id}: {str(e)}", exc_info=True)
        return {"error": f"Internal server error: {str(e)}"}



# Run app (using settings.UVICORN_HOST and settings.UVICORN_PORT)
if __name__ == "__main__":
    uvicorn.run(app, host=settings.UVICORN_HOST, port=settings.UVICORN_PORT)

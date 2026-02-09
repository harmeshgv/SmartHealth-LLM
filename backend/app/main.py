from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.log_level_middleware import PerRouteLogLevelMiddleware
from app.api.v1 import chat, health, debug, metrics
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="SmartHealth API", version="0.1.0", description="SmartHealth backend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home route
@app.get("/")
async def home():
    return {"message": "SmartHealth Backend Running"}


app.add_middleware(PerRouteLogLevelMiddleware)

# Register routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(debug.router)
app.include_router(metrics.router)

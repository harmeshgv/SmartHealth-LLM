from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api import upload

app = FastAPI()

# Serve static Orb files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Include your routes
app.include_router(upload.router)

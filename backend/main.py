from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api import upload

app = FastAPI()

# Include your routes
app.include_router(upload.router)
    
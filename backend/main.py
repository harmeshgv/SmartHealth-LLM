from fastapi import FastAPI
from backend.api import upload
app = FastAPI()

app.include_router(upload.router)
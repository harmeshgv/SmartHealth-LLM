from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.v1 import auth, chat, health

app = FastAPI(
    title="SmartHealth API",
    version="0.1.0",
    description="SmartHealth backend with JWT authentication"
)

# Home route
@app.get("/")
async def home():
    return {"message": "SmartHealth Backend Running"}

# Register routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(health.router)

# ---- CUSTOM OPENAPI: ADD BEARER TOKEN SUPPORT ----
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="SmartHealth API",
        version="0.1.0",
        description="SmartHealth backend with JWT authentication",
        routes=app.routes,
    )

    # Add BearerAuth globally
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    # Force BearerAuth for ALL /chat/* paths
    for path in openapi_schema["paths"]:
        if path.startswith("/chat"):
            for method in openapi_schema["paths"][path]:
                openapi_schema["paths"][path][method]["security"] = [
                    {"BearerAuth": []}
                ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
# ---- END ----

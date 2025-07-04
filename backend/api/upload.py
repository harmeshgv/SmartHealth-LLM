from fastapi import APIRouter, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from backend.services.image_classifier import Classify_Diseases
from backend.services.symptom_to_disease import DiseaseMatcher

router = APIRouter()

classification = Classify_Diseases()
matcher = DiseaseMatcher(vectorstore_path="backend/Vector/symptom_faiss_db")


templates = Jinja2Templates(directory="frontend")

# Store chat history in-memory (reset on server restart)
chat_history = []

@router.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})


@router.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        result, confidence = classification.predict(file.file)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,
            "confidence": confidence,
            "chat_history": chat_history
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat/", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    try:
        matches = matcher.match(user_input, top_k=3)
        if not matches:
            bot_response = "Sorry, I couldn't match any disease to your symptoms."
        else:
            bot_response = "Top possible diseases:\n"
            for i, (disease, score, symptom_text) in enumerate(matches, 1):
                bot_response += f"{i}. {disease} (Confidence: {score:.4f})\n"

        # Append to chat history
        chat_history.append({
            "user": user_input,
            "bot": bot_response
        })

        return templates.TemplateResponse("index.html", {
            "request": request,
            "chat_history": chat_history
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from backend.services.image_classifier import Classify_Diseases
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
router = APIRouter()

classification = Classify_Diseases()

templates = Jinja2Templates(directory="frontend")



@router.get("/", response_class=HTMLResponse)
async def form_page(request : Request):
    return templates.TemplateResponse("index.html",{"request": request})

@router.post("/predict/", response_class=HTMLResponse)
async def predict(request : Request, file : UploadFile = File(...)):
    try:
        result , confidence = classification.predict(file.file)
        return templates.TemplateResponse("index.html", {"request" : request, "result" : result, "confidence" : confidence})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
import os
import pytest

from app.tools.medical.symptom_matcher_tool import SymptomDiseaseMatcherTool
from app.utils.embeddings import EmbeddingSingleton
from app.config import settings


@pytest.mark.asyncio
async def test_symptom_matcher_tool():

    # Ensure FAISS directory exists
    assert os.path.exists(settings.FAISS_SYMPTOM_PATH), "Symptom FAISS index directory missing!"
    assert os.path.exists(os.path.join(settings.FAISS_SYMPTOM_PATH, "index.faiss")), "FAISS file missing!"

    # Load embeddings (singleton)
    embeddings = EmbeddingSingleton.get_instance()

    # Initialize tool
    tool = SymptomDiseaseMatcherTool(
        db_path=settings.FAISS_SYMPTOM_PATH,
        embeddings=embeddings
    )

    # Run tool with common dengue symptoms
    symptoms = ["fever", "joint pain", "headache"]

    result = await tool.run(symptoms, k=3)

    # Validate structure
    assert "matched_diseases" in result, "Result should contain matched_diseases"
    assert isinstance(result["matched_diseases"], list), "matched_diseases must be a list"

    # Validate at least one result
    assert len(result["matched_diseases"]) > 0, "No matches returned from FAISS"

    first = result["matched_diseases"][0]

    # Validate individual match fields
    assert "disease" in first, "Each match must contain 'disease'"
    assert "symptoms" in first, "Each match must contain 'symptoms'"
    assert "score" in first, "Each match must contain 'score'"

    # Optional strong correctness check
    assert isinstance(first["disease"], str)
    assert isinstance(first["symptoms"], str)
    assert isinstance(first["score"], float)



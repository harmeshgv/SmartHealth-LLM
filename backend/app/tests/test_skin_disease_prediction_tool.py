import os
import pytest
from PIL import Image
from app.tools.ml.skin_disease_prediction_tool import SkinDiseasePredictionTool
from app.config import settings
@pytest.mark.asyncio
async def test_skin_disease_predictor():

    model_path = settings.SKIN_MODEL_PATH
    labels_path = settings.SKIN_MODEL_CLASS_PATH

    # Skip test if model does not exist locally
    if not os.path.exists(model_path):
        pytest.skip("Model file missing — skipping test.")

    if not os.path.exists(labels_path):
        pytest.skip("labels.json missing — skipping test.")

    # Create dummy image
    img = Image.new("RGB", (224, 224), color="white")

    tool = SkinDiseasePredictionTool()

    result = await tool.run(img)

    assert "predicted_class" in result
    assert isinstance(result["predicted_class"], str)

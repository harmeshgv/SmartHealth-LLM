# backend/tools/skin_disease_prediction_tool.py
import logging
from typing import Any, Dict, List
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL.Image import Image

from .base import BaseTool

logger = logging.getLogger(__name__)

class SkinDiseasePredictionTool(BaseTool):
    """
    A tool for predicting skin diseases from an image.
    """

    name: str = "Skin Disease Predictor"
    description: str = (
        "Predicts the type of skin disease from a given image. "
        "Input must be a PIL Image object."
    )

    def __init__(self, model_path: str, class_names: List[str]):
        super().__init__()
        self.class_names = class_names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = self._load_model(model_path)
            self.model.eval()
            self.transform = self._get_transform()
            logger.info(f"Skin disease prediction model loaded from {model_path} onto {self.device.upper()}")
        except Exception as e:
            logger.error(f"Failed to load skin disease prediction model: {e}", exc_info=True)
            raise

    def _load_model(self, model_path: str):
        """Loads the pre-trained DenseNet model onto the correct device."""
        model = models.densenet121(weights=None) # pretrained=False is deprecated
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, len(self.class_names))
        # Load the model onto the target device
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device) # Ensure model is on the correct device
        return model

    def _get_transform(self):
        """Returns the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    async def execute(self, image: Image) -> Dict[str, Any]:
        """
        Predicts the disease from a PIL image.

        Args:
            image: A PIL Image object to be classified.

        Returns:
            A dictionary containing the predicted class or an error.
        """
        if not isinstance(image, Image):
            return {"error": "Input must be a PIL Image object."}

        try:
            # Transform image and move tensor to the correct device
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, preds = torch.max(outputs, 1)

            predicted_class = self.class_names[preds.item()]
            logger.info(f"Predicted skin disease: {predicted_class}")
            return {"predicted_class": predicted_class}
        except Exception as e:
            error_msg = f"An error occurred during image prediction: {e}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

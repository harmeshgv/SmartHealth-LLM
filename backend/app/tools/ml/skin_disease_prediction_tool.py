import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL.Image import Image
from typing import Dict, Any, List
from app.tools.base_tool import BaseTool
from app.config import settings
import os


class SkinDiseasePredictionTool(BaseTool):

    name = "skin_disease_predictor"
    description = "Predicts skin disease class from a given image."

    def __init__(self):
        super().__init__()

        self.model_path = settings.SKIN_MODEL_PATH
        self.labels_path = settings.SKIN_MODEL_CLASS_PATH

        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"labels.json missing at: {self.labels_path}")

        # Load class list from labels.json
        with open(self.labels_path, "r") as f:
            self.class_names: List[str] = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = self._load_model()
            self.model.eval()
            self.transform = self._build_transforms()


        except Exception as e:
            raise RuntimeError(f"Failed to load skin model: {e}")

    # --------------------------------------------------------
    def _load_model(self):
        """Load DenseNet model and adjust classifier for class count."""
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, len(self.class_names))

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    # --------------------------------------------------------
    def _build_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    # --------------------------------------------------------
    async def run(self, image: Image) -> Dict[str, Any]:
        if not isinstance(image, Image):
            return {"error": "Input must be a PIL Image"}

        try:
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                _, pred_idx = torch.max(outputs, 1)

            predicted = self.class_names[pred_idx.item()]
            return {"predicted_class": predicted}

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

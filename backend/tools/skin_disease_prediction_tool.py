import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from config import DISEASE_PREDICTION_MODEL  # ✅ Correct import


class SkinDiseasePrediction:
    def __init__(
        self, model_path: str = DISEASE_PREDICTION_MODEL
    ):  # ✅ Use correct model path
        self.model = models.densenet121(pretrained=False)
        self.num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_features, 8)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )  # ✅ Now loads .pth file
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_image(self, image_path, class_names):
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, preds = torch.max(outputs, 1)

        predicted_class = class_names[preds.item()]
        return predicted_class

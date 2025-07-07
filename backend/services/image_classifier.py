import tensorflow as tf
import json
import numpy as np
from backend.utils.image_preprocessing import PreProcessor
from backend.config import LABELS_JSON

class Classify_Diseases:
    def __init__(self, model_path = "backend/models/skin_disease_model.h5", labels_path = LABELS_JSON):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = self._load_model()
        self.classnames = self._load_class_names()
        self.preprocessor = PreProcessor()
    
    def _load_model(self):
        return tf.keras.models.load_model(self.model_path)
        
    def _load_class_names(self):
        with open(self.labels_path) as f:
            return json.load(f)
        
    def predict(self, image_file):
        img_array = self.preprocessor.preprocess_image(image_file)
        preds = self.model.predict(img_array)
        predicted_index = np.argmax(preds)
        predicted_label = self.classnames[predicted_index]
        confidence = float(np.max(preds))
        return predicted_label, confidence
    

# filtering_with_ner.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

class RemoveUselessWords:
    
    def __init__(self):
        self.model_name = "d4data/biomedical-ner-all"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer, 
                          aggregation_strategy="simple")
        
        self.allowed_entities = [
            "Sign_symptom", "Disease_disorder", "Biological_structure",
            "Medication", "Therapeutic_procedure", "Duration",
            "Diagnostic_procedure", "Body_part", "Medical_device",
            "Lab_value", "Frequency", "Severity_value"
        ]
    
    def preprocess_for_ner(self, text: str) -> str:
        """Prepares text for NER processing"""
        # Fix concatenated words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)(\w)', r'\1 \2', text)
        return text
    
    def process_entities(self, text: str) -> list:
        """Extracts medical entities from text"""
        text = self.preprocess_for_ner(text)
        entities = self.ner(text)
        return [
            ent["word"].strip().lower()
            for ent in entities
            if ent["entity_group"] in self.allowed_entities
        ]
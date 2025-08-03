# filtering_with_ner.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import spacy


class DataPreprocessing:
    
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
        self.nlp = spacy.load("en_core_web_sm")

    
    def _cleaning(self, text: str) -> str:
        """Prepares text for NER processing"""
        # Fix concatenated words
        text = re.sub(r"[.,:]", " ", text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)(\w)', r'\1 \2', text)

        doc = self.nlp(text)

        tokens =  [
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop and not token.like_num and not token.is_punct
        ]
        return " ".join(tokens)
    
    def _filtering(self, text: str) -> list:
        entities = self.ner(text)

        return [
            ent["word"].strip().lower()
            for ent in entities
            if ent["entity_group"] in self.allowed_entities
        ]    


    def preprocess(self, text) -> str:

        if not isinstance(text, str):  # Safe guard
            text = ""

        text = self._cleaning(text)
        return self._filtering(text)
    

        
        

    
    

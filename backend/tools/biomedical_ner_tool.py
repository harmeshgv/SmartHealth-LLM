# backend/tools/biomedical_ner_tool.py
import os

# use temp directory for cache (always writable on Spaces)
cache_dir = "/tmp/hf_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

print(f"Hugging Face cache directory set to: {cache_dir}")


from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict
import numpy as np



class BiomedicalNER:
    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="average",
        )

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract biomedical entities from text.

        Args:
            text (str): Input biomedical text

        Returns:
            List[Dict]: List of extracted entities with start/end positions and labels
        """
        try:
            entities = self.pipe(text)
            return entities
        except Exception as e:
            return []


# Instantiate the model
ner = BiomedicalNER()


def extract_data(input_dict: dict) -> dict:
    """
    Extract biomedical entities and return context/entity types instead of exact names.
    """
    text = input_dict.get("input", "")
    entities = ner.extract_entities(text)

    # Convert any float32 scores to Python float
    for e in entities:
        for key in e:
            if isinstance(e[key], np.float32):
                e[key] = float(e[key])

    # Extract entity types/groups instead of exact names
    entity_types = []
    entity_context = []

    for entity in entities:
        entity_group = entity.get("entity_group", "").lower()
        entity_word = entity.get("word", "").strip()

        if entity_group and entity_word:
            # Categorize entity types
            if entity_group.startswith("disease"):
                entity_types.append("disease_condition")
                entity_context.append(f"disease/condition: {entity_word}")
            elif entity_group.startswith("drug"):
                entity_types.append("drug_treatment")
                entity_context.append(f"drug/treatment: {entity_word}")
            elif entity_group.startswith("symptom"):
                entity_types.append("symptom")
                entity_context.append(f"symptom: {entity_word}")
            elif entity_group.startswith("anatomy"):
                entity_types.append("body_part")
                entity_context.append(f"body part: {entity_word}")
            elif entity_group.startswith("chemical"):
                entity_types.append("chemical")
                entity_context.append(f"chemical: {entity_word}")
            else:
                entity_types.append(entity_group)
                entity_context.append(f"{entity_group}: {entity_word}")

    # Remove duplicates while preserving order
    entity_types = list(dict.fromkeys(entity_types))
    entity_context = list(dict.fromkeys(entity_context))

    # Determine primary context for routing
    primary_context = None
    if "disease_condition" in entity_types:
        primary_context = "disease_condition"
    elif "symptom" in entity_types:
        primary_context = "symptom"
    elif "drug_treatment" in entity_types:
        primary_context = "drug_treatment"
    elif entity_types:
        primary_context = entity_types[0]
    else:
        primary_context = "general_medical"

    return {
        "input": text,
        "entities": entities,
        "entity_types": entity_types,
        "entity_context": entity_context,
        "primary_context": primary_context,
        "has_medical_context": len(entity_types) > 0,
    }


# Test run
if __name__ == "__main__":
    sample_texts = [
        "The patient was diagnosed with diabetes and prescribed metformin.",
        "I have a headache and fever.",
        "My knee pain is getting worse.",
        "What is aspirin used for?",
        "Tell me about common cold symptoms.",
    ]

    for sample_text in sample_texts:
        output = extract_data({"input": sample_text})
        print(f"Input: {sample_text}")
        print(f"Entity Types: {output['entity_types']}")
        print(f"Primary Context: {output['primary_context']}")
        print(f"Entity Context: {output['entity_context']}")
        print(f"Has Medical Context: {output['has_medical_context']}")
        print("-" * 50)

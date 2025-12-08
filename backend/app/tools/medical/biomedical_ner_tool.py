from app.tools.base_tool import BaseTool
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class BiomedicalNERTool(BaseTool):
    name = "biomedical_ner"
    description = "Extracts biomedical entities (symptoms, diseases) using HF NER model."

    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        self.pipe = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="average"
        )

    async def run(self, text: str):
        raw = self.pipe(text)

        # Output format:
        # [{'word': 'fever', 'entity_group': 'SYMPTOM'}, ...]

        entities = [
            {"text": item["word"], "type": item["entity_group"]}
            for item in raw
        ]

        # Organize by type
        result = {
            "symptoms": [e["text"] for e in entities if e["type"] in ["SYMPTOM"]],
            "diseases": [e["text"] for e in entities if e["type"] in ["DISEASE"]],
            "drugs": [e["text"] for e in entities if e["type"] in ["DRUG"]],
            "entities": entities  # full detail if needed
        }

        return result

import time
import logging
from app.tools.base_tool import BaseTool
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logger = logging.getLogger(__name__)


class BiomedicalNERTool(BaseTool):
    name = "biomedical_ner"
    description = "Extracts biomedical entities (symptoms, diseases) using HF NER model."

    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        super().__init__()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)

            self.pipe = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="average"
            )
        except Exception as e:
            logger.error(f"Failed to load Biomedical NER tool: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Biomedical NER tool: {e}")

    async def run(self, text: str, run_id: str = None):
        start_time = time.time()
        logger.info("Tool run started", extra={"run_id": run_id, "tool": self.name})

        if not text or not isinstance(text, str):
            logger.warning("Input text must be a non-empty string.", extra={"run_id": run_id, "tool": self.name})
            return {"error": "Input text must be a non-empty string."}

        try:
            raw = self.pipe(text)
        except Exception as e:
            logger.error(f"Error during NER pipeline execution: {e}", extra={"run_id": run_id, "tool": self.name}, exc_info=True)
            return {"error": f"NER pipeline execution failed: {e}"}

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

        end_time = time.time()
        latency = end_time - start_time
        logger.info("Tool run finished", extra={"run_id": run_id, "tool": self.name, "latency": latency})

        return result

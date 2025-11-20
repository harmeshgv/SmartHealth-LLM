# backend/tools/biomedical_ner_tool.py
import logging
import os
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np

from .base import BaseTool

logger = logging.getLogger(__name__)

# Note: Caching setup should be handled at the application's entry point,
# not as a side effect of importing a module.
# Consider setting environment variables like HF_HOME and TRANSFORMERS_CACHE
# before the application starts.

class BiomedicalNERTool(BaseTool):
    """
    A tool for extracting biomedical entities from text using a Hugging Face model.
    """

    name: str = "Biomedical NER"
    description: str = (
        "Extracts biomedical entities (like diseases, symptoms, drugs) from a given text. "
        "Input should be a string of text."
    )

    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        super().__init__()
        try:
            # Configure caching. This is better handled by global environment variables
            # but can be set here if needed for this specific tool.
            cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf_cache")
            os.makedirs(cache_dir, exist_ok=True)

            logger.info(f"Using Hugging Face cache directory: {cache_dir}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
            self.pipe = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="average",
            )
            logger.info(f"Biomedical NER model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Biomedical NER model '{model_name}': {e}", exc_info=True)
            raise

    async def execute(self, text: str) -> Dict[str, Any]:
        """
        Extracts biomedical entities from the given text.

        Args:
            text: The input text.

        Returns:
            A dictionary containing the list of extracted entities or an error.
        """
        if not text or not isinstance(text, str):
            return {"error": "Input must be a non-empty string."}

        try:
            entities = self.pipe(text)

            # Sanitize numpy types for JSON serialization
            for entity in entities:
                for key, value in entity.items():
                    if isinstance(value, np.float32):
                        entity[key] = float(value)

            logger.info(f"Extracted {len(entities)} entities from text.")
            return {"entities": entities}
        except Exception as e:
            error_msg = f"An error occurred during NER entity extraction: {e}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

# NOTE TO THE DEVELOPER:
# The complex logic for categorizing entities and determining 'primary_context'
# has been removed from this tool. A tool should be simple and do one thing well.
# This logic is business logic that should be owned by an *agent*.
# The agent can use this tool to get the raw entities, and then perform its
# own analysis and decision-making based on the results.

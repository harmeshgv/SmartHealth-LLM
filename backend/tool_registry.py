    # backend/tool_registry.py
import logging
from typing import Dict, Any

from .config import settings
from .utils.embeddings import get_embeddings # Assuming get_embeddings is correctly implemented
from .tools import (
    BiomedicalNERTool,
    DiseaseInfoRetrieverTool,
    SymptomDiseaseMatcherTool,
    GoogleSearchTool,
    SkinDiseasePredictionTool,
)

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    A central registry to initialize and hold instances of all available tools.
    This ensures that tools are instantiated once and their dependencies are
    correctly injected from the centralized settings.
    """
    def __init__(self):
        logger.info("Initializing Tool Registry...")

        # Initialize embeddings only if needed by any tool
        self.embeddings_instance = None
        if settings.SYMPTOM_FAISS_DB: # Check if a tool relies on embeddings
             self.embeddings_instance = get_embeddings()

        self.tools: Dict[str, Any] = {} # Store tool instances
        self._initialize_tools()

        logger.info(f"Initialized {len(self.tools)} tools: {list(self.tools.keys())}")

    def _initialize_tools(self):
        """Internal method to instantiate all tools."""
        try:
            self.tools["biomedical_ner_tool"] = BiomedicalNERTool(
                model_name=settings.BIOMEDICAL_NER_MODEL_NAME
            )
            logger.debug("BiomedicalNERTool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize BiomedicalNERTool: {e}")

        try:
            if self.embeddings_instance:
                self.tools["disease_info_retriever_tool"] = DiseaseInfoRetrieverTool(
                    csv_path=settings.MAYO_CSV,
                    embeddings=self.embeddings_instance
                )
                logger.debug("DiseaseInfoRetrieverTool initialized.")
            else:
                logger.warning("Embeddings instance not available, DiseaseInfoRetrieverTool skipped.")
        except Exception as e:
            logger.error(f"Failed to initialize DiseaseInfoRetrieverTool: {e}")

        try:
            if self.embeddings_instance:
                self.tools["symptom_disease_matcher_tool"] = SymptomDiseaseMatcherTool(
                    db_path=settings.SYMPTOM_FAISS_DB,
                    embeddings=self.embeddings_instance
                )
                logger.debug("SymptomDiseaseMatcherTool initialized.")
            else:
                logger.warning("Embeddings instance not available, SymptomDiseaseMatcherTool skipped.")
        except Exception as e:
            logger.error(f"Failed to initialize SymptomDiseaseMatcherTool: {e}")

        try:
            self.tools["google_search_tool"] = GoogleSearchTool()
            logger.debug("GoogleSearchTool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize GoogleSearchTool: {e}")

        try:
            self.tools["skin_disease_prediction_tool"] = SkinDiseasePredictionTool(
                model_path=settings.DISEASE_PREDICTION_MODEL,
                class_names=settings.SKIN_DISEASE_CLASS_NAMES
            )
            logger.debug("SkinDiseasePredictionTool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize SkinDiseasePredictionTool: {e}")

    def get_tool(self, name: str):
        """
        Retrieves a tool instance by its registered name.

        Args:
            name: The key under which the tool was registered in the registry.

        Returns:
            The instantiated tool object, or None if not found.
        """
        return self.tools.get(name)

# Create a single, shared instance of the registry
tool_registry = ToolRegistry()

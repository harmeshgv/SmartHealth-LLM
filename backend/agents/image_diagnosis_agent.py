import logging
from typing import Any, Dict
from PIL.Image import Image as PilImage

from .base_agent import BaseAgent
from ..tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

class ImageDiagnosisAgent(BaseAgent):
    name: str = "Image Diagnosis Agent"
    description: str = "An agent that uses an ML tool to predict a skin disease from an image."

    def __init__(self, tool_registry: ToolRegistry, **kwargs: Any):
        # This agent doesn't need an LLM, so we don't pass it to super()
        super().__init__()
        self.tool_registry = tool_registry
        logger.info("ImageDiagnosisAgent initialized.")

    async def invoke(self, state: Dict[str, Any], **kwargs: Any) -> dict:
        """
        The main entry point for the ImageDiagnosisAgent.
        It uses the skin disease prediction tool to predict a disease from an image.
        """
        image: PilImage = kwargs.get("image", state.get("image"))
        if not image:
            logger.warning("ImageDiagnosisAgent invoked without an image.")
            return {"disease_name": "unknown", "raw_result": "No image provided to agent."}
            
        logger.info("ImageDiagnosisAgent invoked with an image.")
        
        skin_predictor_tool = self.tool_registry.get_tool("skin_disease_prediction_tool")
        if not skin_predictor_tool:
            logger.error("Skin disease prediction tool not found in registry.")
            return {"disease_name": "unknown", "raw_result": "Error: Skin disease prediction tool not available."}
            
        try:
            prediction_result = await skin_predictor_tool.execute(image=image) # Assuming tool.execute is async
            predicted_class = prediction_result.get("predicted_class", "unknown skin condition")
            logger.info(f"ImageDiagnosisAgent predicted disease: {predicted_class}")

            # Return the result to be merged into the main orchestration state
            return {
                "disease_name": predicted_class,
                "decision": "image_analysis", # Signal the decision made
            }
        except Exception as e:
            logger.error(f"Error during image prediction in ImageDiagnosisAgent: {e}", exc_info=True)
            return {
                "disease_name": "unknown skin condition",
                "raw_result": f"An error occurred during image analysis: {e}",
            }

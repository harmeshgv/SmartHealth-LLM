# backend/tools/google_search.py
import logging
from typing import Any, Dict
from langchain_community.utilities import GoogleSerperAPIWrapper

from .base import BaseTool
from ..config import settings

logger = logging.getLogger(__name__)

class GoogleSearchTool(BaseTool):
    """A tool for performing Google searches using the Serper API."""

    name: str = "Google Search"
    description: str = (
        "Retrieves information from Google search. "
        "Input should be a search query string."
    )

    def __init__(self):
        super().__init__()
        if not settings.SERPER_API_KEY:
            raise ValueError("SERPER_API_KEY is not set in the environment or .env file.")
        self.search = GoogleSerperAPIWrapper(serper_api_key=settings.SERPER_API_KEY)

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Executes the Google search.

        Args:
            query: The search query string.

        Returns:
            A dictionary containing the search results or an error.
        """
        if not query or not isinstance(query, str):
            error_msg = "Invalid query. Please provide a non-empty string."
            logger.error(error_msg)
            return {"error": error_msg}

        try:
            result = self.search.run(query)
            if not result:
                return {"result": "No results found."}
            return {"result": result}
        except Exception as e:
            error_msg = f"An error occurred during the search: {e}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

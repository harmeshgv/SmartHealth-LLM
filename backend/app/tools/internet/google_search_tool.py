from typing import Any, Dict
from langchain_community.utilities import GoogleSerperAPIWrapper

from app.tools.base_tool import BaseTool
from app.config import settings


class GoogleSearchTool(BaseTool):
    name = "google_search"
    description = "Searches Google via Serper API using LangChain wrapper."

    def __init__(self):
        super().__init__()

        if not settings.SERPER_API_KEY:
            raise ValueError("SERPER_API_KEY is missing in environment variables.")

        # LangChain Serper wrapper (sync)
        self.search = GoogleSerperAPIWrapper(
            serper_api_key=settings.SERPER_API_KEY
        )

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Execute a Google search and return the result.
        """
        if not query or not isinstance(query, str):
            return {"error": "Invalid query. Must be a non-empty string."}

        try:
            # LangChain wrapper is sync → works inside async
            result = self.search.run(query)

            if not result:
                return {"result": "No results found."}

            return {"result": result}
        except Exception as e:
            return {"error": f"Error during Google search: {e}"}

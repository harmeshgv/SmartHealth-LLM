import time
import logging
from typing import Any, Dict
from langchain_community.utilities import GoogleSerperAPIWrapper

from app.tools.base_tool import BaseTool
from app.config import settings
from app.core.metrics_tracker import metrics_tracker

logger = logging.getLogger(__name__)


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

    async def run(self, query: str, run_id: str = None) -> Dict[str, Any]:
        """
        Execute a Google search and return the result.
        """
        start_time = time.time()
        logger.info("Tool run started", extra={"run_id": run_id, "tool": self.name})

        if not query or not isinstance(query, str):
            logger.warning("Invalid query. Must be a non-empty string.", extra={"run_id": run_id, "tool": self.name})
            metrics_tracker.record_tool_event(
                run_id=run_id, tool_name=self.name, source="internet", success=False, error=True
            )
            return {"error": "Invalid query. Must be a non-empty string."}

        logger.debug(f"Executing Google search with query: {query}", extra={"run_id": run_id, "tool": self.name, "query": query})
        try:
            # LangChain wrapper is sync → works inside async
            result = self.search.run(query)
            logger.debug(f"Raw result from Google search: {result}", extra={"run_id": run_id, "tool": self.name, "raw_result": result})

            if not result:
                output = {"result": "No results found."}
            else:
                output = {"result": result}

            end_time = time.time()
            latency = end_time - start_time
            logger.info("Tool run finished", extra={"run_id": run_id, "tool": self.name, "latency": latency})
            metrics_tracker.record_tool_event(
                run_id=run_id, tool_name=self.name, source="internet", success=True
            )

            return output
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error("Tool run failed", extra={"run_id": run_id, "tool": self.name, "latency": latency, "error": str(e)})
            metrics_tracker.record_tool_event(
                run_id=run_id, tool_name=self.name, source="internet", success=False, error=True
            )
            return {"error": f"Error during Google search: {e}"}

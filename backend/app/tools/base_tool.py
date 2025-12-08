from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Base class for all tools in the Multi-Agent System.

    Every tool must implement:
        - name (string)
        - description (string)
        - run() (async)
    """

    # Tool ID used in registry
    name: str = "base_tool"

    # Human-readable explanation
    description: str = "Base tool interface"

    # Optional config
    config: Dict[str, Any] = {}

    def __init__(self):
        """Common init for all tools."""
        pass

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """
        Main execution method for the tool.
        Must be implemented by all subclasses.
        """
        raise NotImplementedError("Tool must implement run() method.")

    def info(self) -> Dict[str, str]:
        """
        Returns metadata about this tool.
        Useful for LLM (Decider/Planner) to know what tool does.
        """
        return {
            "name": self.name,
            "description": self.description,
        }

    def __repr__(self):
        return f"<Tool name={self.name}>"

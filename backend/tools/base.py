# backend/tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    """
    Abstract base class for all tools in the system.

    This class defines a common interface that all tools must implement.
    This ensures that all tools are consistent, predictable, and interchangeable.
    """

    # You can add a name and description for each tool
    name: str = "Base Tool"
    description: str = "This is a template for a tool."

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        The main execution method for the tool.

        Subclasses must implement this method. It takes a flexible number
        of keyword arguments and should always return a dictionary.

        Args:
            **kwargs: A dictionary of arguments required by the tool.

        Returns:
            A dictionary containing the result of the tool's execution.
        """
        pass

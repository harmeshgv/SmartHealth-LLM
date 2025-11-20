from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    name: str = "Base Agent"
    description: str = "This is a template for an agent."

    def __init__(self, llm_instance: Any = None):
        self.llm = llm_instance

    @abstractmethod
    async def invoke(self, state: Dict[str, Any], **kwargs: Any) -> Any:
        """
        Invoke the agent with the given state and additional parameters.
        All agents must implement this asynchronous method.

        Args:
            state: The current state of the agent orchestration.
            **kwargs: Additional parameters specific to the agent's invocation.

        Returns:
            The result of the agent's operation, to be incorporated back into the state.
        """
        pass

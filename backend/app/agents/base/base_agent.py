import logging
from abc import ABC, abstractmethod
from app.llm.factories.llm_factory import LLMFactory
from app.core.metrics_tracker import metrics_tracker

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    agent_name = "base_agent"

    def __init__(self, context):
        self.context = context
        self.session_id = context.session_id

        self.llm = LLMFactory.for_agent(self.agent_name)

        self.tools = context.tools
        logger.debug(
            "Agent initiallized | agent=%s | session_id=%s",
            self.agent_name,
            self.session_id,
        )

    async def llm_call(self, prompt: str, run_id: str, **kwargs):
        logger.info(
            "LLM call Started",
            extra={"agent": self.agent_name, "run_id": run_id},
        )

        try:
            response = await self.llm.generate(prompt, **kwargs)
            return response
        except Exception:
            logger.error(
                "LLM call Failed",
                extra={"agent": self.agent_name, "run_id": run_id},
            )
            raise

    async def save_memory(self, user_message: str, llm_output: str, run_id: str):
        try:
            await self.context.short_memory.save(
                session_id=self.session_id,
                user_message=user_message,
                agent_output=llm_output,
            )

            await self.context.long_memory.save(
                session_id=self.session_id,
                user_message=user_message,
                agent_output=llm_output,
            )
            metrics_tracker.record_memory_save(run_id=run_id)

            logger.info(
                "Memory saved",
                extra={"agent": self.agent_name, "run_id": run_id},
            )
        except Exception:
            logger.error(
                "Memory save Failed",
                extra={"agent": self.agent_name, "run_id": run_id},
            )

    async def recall_memory(self, run_id: str, n=5):
        """
        Recall last N messages from short-term memory.
        """
        logger.info("Recalling Memory", extra={"run_id": run_id, "last_n": n})

        past = await self.context.long_memory.get(self.session_id)
        recalled = past[-n:]
        metrics_tracker.record_memory_recall(run_id=run_id, items_used=len(recalled))
        return recalled

    @abstractmethod
    async def run(self, *args, run_id: str, **kwargs):
        pass

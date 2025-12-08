from abc import ABC, abstractmethod
from app.llm.factories.llm_factory import LLMFactory

class BaseAgent(ABC):
    agent_name = "base_agent"

    def __init__(self, context):
        self.context = context
        self.session_id = context.session_id

        self.llm = LLMFactory.for_agent(self.agent_name)

        self.tools = context.tools
        self.debug = context.debug


    async def llm_call(self, prompt: str, **kwargs):
        if self.debug:
            print(f"[DEBUG] LLM prompt for {self.agent_name}:\n{prompt}")

        response = await self.llm.generate(prompt, **kwargs)

        if self.debug:
            print(f"[DEBUG] LLM response for {self.agent_name}:\n{response}")

        return response


    async def save_memory(self, user_message: str, llm_output: str):
        await self.context.short_memory.save(
            session_id=self.session_id,
            user_message=user_message,
            agent_output=llm_output
        )

        await self.context.long_memory.save(
            session_id=self.session_id,
            user_message=user_message,
            agent_output=llm_output
        )



    async def recall_memory(self, n=5):
        """
        Recall last N messages from short-term memory.
        """
        past = await self.context.long_memory.get(self.session_id)
        return past[-n:]


    @abstractmethod
    async def run(self, *args, **kwargs):
        pass

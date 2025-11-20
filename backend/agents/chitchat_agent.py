import logging
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ChitChatAgent(BaseAgent):
    name: str = "ChitChat Agent"
    description: str = "Handles casual, non-medical conversation and small talk."

    def __init__(self, llm: Any):
        super().__init__(llm_instance=llm)
        logger.info("ChitChatAgent initialized.")

        self.system_prompt = """
        You are a friendly, conversational, and helpful AI medical assistant for a healthcare app called SmartHealth.

        Your behavior rules:
        1. If the user greets you, greet them back warmly.
        2. If the user asks how you are, respond positively.
        3. If the user thanks you, reply politely.
        4. If the user asks what you can do, explain briefly that you assist with:
        - symptom checking
        - disease information
        - analyzing medical images
        - general medical guidance
        5. Keep responses concise, natural, and supportive.
        6. If the user asks about anything NOT related to medical, health, or your defined abilities:
        - Politely refuse.
        - Say that you can only help with medical-related topics.
        - Redirect the user back to health-related assistance.
        """


        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}"),
            ]
        )
        self.chain = self.prompt_template | self.llm

    async def invoke(self, state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Generates a conversational response for non-medical queries.
        This agent's response is final and does not need further formatting.
        """
        logger.debug(f"ChitChatAgent invoked with state: {state}")

        current_state = {
            "query": state.get("query", ""),
            "chat_history": state.get("chat_history", []),
            **kwargs,
        }

        response = await self.chain.ainvoke(current_state)
        logger.info("ChitChatAgent successfully generated a response.")

        # This agent's output is the final, formatted result.
        return {"formatted_result": response.content.strip()}

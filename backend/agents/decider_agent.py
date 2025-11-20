import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable
from typing import Any, Dict

from .base_agent import BaseAgent # Import BaseAgent

logger = logging.getLogger(__name__)

class DeciderAgent(BaseAgent): # Inherit from BaseAgent
    name: str = "Decider Agent"
    description: str = "Analyzes user query and conversation history to decide which specialized agent should handle it."

    def __init__(self, llm: Any): # llm is passed during instantiation
        super().__init__(llm_instance=llm) # Pass llm to BaseAgent
        logger.info("DeciderAgent initialized.")

        self.system_prompt = """
            You are a decision-making agent for a medical assistant system.
            Analyze the user's most recent query in the context of the conversation history and decide which of the following categories it falls into:

            - "symptom_to_disease": The user is describing symptoms (e.g., "I have a fever," "I feel nauseous").
            - "disease_info": The user is asking for information about a specific disease, condition, or medical concept (e.g., "What is diabetes?", "Tell me about migraines").
            - "image_analysis": The user has uploaded an image and is asking for a diagnosis or information related to it.
            - "chitchat": The user is making small talk, asking a general question, greeting you, or thanking you (e.g., "Hello", "Thanks!", "How are you?", "What can you do?").

            Return ONLY one of these four strings: "symptom_to_disease", "disease_info", "image_analysis", or "chitchat".

            Examples based on the last user message:
            User: "I have fever and cough" → "symptom_to_disease"
            User: "Tell me about diabetes" → "disease_info"
            User: "thanks for the help" → "chitchat"
            User: "hi there" → "chitchat"
            User: [sends an image of a rash] "What could this be?" -> "image_analysis"
            User: "What are your capabilities?" -> "chitchat"
            """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}")
            ]
        )

        self.agent: RunnableSerializable = self.prompt_template | self.llm

    async def invoke(self, state: Dict[str, Any], **kwargs: Any) -> str: # Make invoke async
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])

        if not query:
            logger.warning("DeciderAgent received empty query.")
            return "chitchat" # Default to chitchat if query is empty

        logger.debug(f"DeciderAgent invoking with query: {query} and history of {len(chat_history)} messages.")
        
        response = await self.agent.ainvoke({"query": query, "chat_history": chat_history})
        decision = response.content.strip().lower()

        # Normalize the response
        if "symptom" in decision:
            final_decision = "symptom_to_disease"
        elif "disease" in decision:
            final_decision = "disease_info"
        elif "image" in decision:
            final_decision = "image_analysis"
        elif "chitchat" in decision or "chat" in decision:
            final_decision = "chitchat"
        else:
            final_decision = "chitchat" # Default fallback to be safe

        logger.info(f"DeciderAgent decision for query '{query}': {final_decision}")
        return final_decision

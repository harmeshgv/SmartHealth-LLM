from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain.agents import initialize_agent


class DECIDERAGENT:
    def __init__(self, llm):
        self.llm = llm

        self.system_prompt = """
You are a decision-making agent.
Tools available:
- symptom_to_disease: Takes a list of symptoms and returns the most likely disease.
- disease_info: Takes a disease name and returns information about it.

Decide the BEST tool to use and extract the parameters.
Examples:
User: I have fever and cough
Agent: symptom_to_disease
User: Tell me about diabetes
Agent: disease_info

Return STRICTLY the agent name as a string, nothing else.
"""
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{query}")]
        )

        # Just chain the prompt to the LLM
        self.agent: RunnableSerializable = self.prompt_template | self.llm

    def invoke(self, query: str) -> str:
        response = self.agent.invoke({"query": query})
        return response

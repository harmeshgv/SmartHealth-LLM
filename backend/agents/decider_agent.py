from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable


class DeciderAgent:
    def __init__(self, llm):
        self.llm = llm

        self.system_prompt = """
            You are a decision-making agent for a medical assistant system.

            Analyze the user's query and decide which agent should handle it:

            - Use "symptom_to_disease" if the user describes symptoms and wants to know what disease they might have.
            - Use "disease_info" if the user asks for information about a specific disease.

            Return ONLY one of these two strings: "symptom_to_disease" or "disease_info"

            Examples:
            User: "I have fever and cough" → "symptom_to_disease"
            User: "Tell me about diabetes" → "disease_info"
            User: "What are the symptoms of COVID?" → "disease_info"
            User: "I'm experiencing headache and nausea" → "symptom_to_disease"
            """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{query}")]
        )

        self.agent: RunnableSerializable = self.prompt_template | self.llm

    def invoke(self, query: str) -> str:
        response = self.agent.invoke({"query": query})
        decision = response.content.strip().lower()

        # Normalize the response
        if "symptom" in decision:
            return "symptom_to_disease"
        else:
            return "disease_info"

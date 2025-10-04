from langchain.prompts import ChatPromptTemplate


class SymptomToDiseaseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
        You are a medical assistant. Your ONLY task is to identify the SINGLE most likely disease based on the given symptoms.

        CRITICAL RULES:
        - Return ONLY the disease name as plain text
        - NO explanations, NO probabilities, NO multiple diseases
        - JUST the disease name, nothing else

        Examples:
        Input: "I have fever, cough, and cold" → Output: "Influenza"
        Input: "Headache and runny nose" → Output: "Common Cold"
        """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{symptoms}")]
        )
        self.chain = self.prompt_template | self.llm

    def invoke(self, query: str) -> str:
        """Extract ONLY disease name from symptoms"""
        response = self.chain.invoke({"symptoms": query})
        return response.content.strip()

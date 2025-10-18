from langchain_core.prompts import ChatPromptTemplate


class DiseaseInfoAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
        You are a medical information assistant. Provide comprehensive disease information.

        INSTRUCTIONS:
        1. Extract disease name from the query if needed
        2. Provide relevant medical information
        3. Return raw information with clear headings
        4. No formatting, no extra text - just the information

        Examples:
        Input: "Influenza" → Output: "Symptoms: Fever, cough... Causes: Viral infection..."
        Input: "symptoms of malaria" → Output: "Symptoms: High fever, chills..."
        """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{query}")]
        )
        self.chain = self.prompt_template | self.llm

    def invoke(self, query: str) -> str:
        """Get raw disease information"""
        response = self.chain.invoke({"query": query})
        return response.content.strip()

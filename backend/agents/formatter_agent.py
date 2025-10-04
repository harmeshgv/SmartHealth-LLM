from langchain.prompts import ChatPromptTemplate


class FormatterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
        You are a medical response formatter. Your task is to format the raw medical information into a clean, user-friendly response.

        You receive:
        - Original user query
        - Disease name (if available)
        - Raw medical information
        - Decision type (symptom_to_disease or disease_info)

        Formatting Rules:
        1. For symptom queries (symptom_to_disease):
           - Start with: "Based on your symptoms, you might have: [Disease Name]"
           - Then provide the information in a structured way
           - Use clear sections like: Symptoms, Causes, Treatment, Prevention
           - Make it empathetic and easy to understand

        2. For direct disease queries (disease_info):
           - Start with: "Here's information about [Disease Name]:"
           - Organize the information based on what the user asked for
           - If user asked for specific info (like symptoms/treatment), focus on that
           - Use clear headings and bullet points

        3. Always include:
           - Clear section headings
           - Easy-to-read format
           - Professional but friendly tone
           - Important warnings if any (like "Consult a doctor for proper diagnosis")

        Example Input:
        Query: "I have fever and cough"
        Disease Name: "Influenza"
        Raw Result: "Symptoms: Fever, cough, body aches... Causes: Viral infection..."
        Decision: "symptom_to_disease"

        Example Output:
        "Based on your symptoms, you might have: Influenza

        🤒 Symptoms:
        • Fever
        • Cough
        • Body aches
        • Fatigue

        🦠 Causes:
        • Viral infection

        💊 Treatment:
        • Rest
        • Plenty of fluids
        • Over-the-counter fever reducers

        ⚠️ Please consult a healthcare professional for proper diagnosis and treatment."
        """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """
            Original Query: {query}
            Disease Name: {disease_name}
            Raw Medical Information: {raw_result}
            Decision Type: {decision}

            Please format this into a clean, user-friendly response:
            """,
                ),
            ]
        )
        self.chain = self.prompt_template | self.llm

    def invoke(
        self,
        query: str,
        disease_name: str = "",
        raw_result: str = "",
        decision: str = "",
    ) -> str:
        """Format all the accumulated information into a clean response"""
        response = self.chain.invoke(
            {
                "query": query,
                "disease_name": disease_name,
                "raw_result": raw_result,
                "decision": decision,
            }
        )
        return response.content.strip()

from langchain_core.prompts import ChatPromptTemplate


# Update your FormatterAgent class to accept the decision parameter
class FormatterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
        You are a medical response formatter. Your task is to transform complex medical information into short, easy-to-understand, point-wise responses.

        RULES:
        1. KEEP IT SHORT & SIMPLE - Maximum 8-10 key points total
        2. USE EMOJIS for visual appeal
        3. USE PLAIN LANGUAGE - explain medical terms in simple words
        4. POINT-WISE FORMAT - no long paragraphs
        5. FOCUS ON KEY INFORMATION - skip minor details
        6. FRIENDLY & REASSURING tone

        FORMATTING GUIDELINES:

        For symptom queries (symptom_to_disease):
        🔍 Based on your symptoms, this could be: [Disease Name]

        📋 Main Symptoms:
        • [Symptom 1] - in simple terms
        • [Symptom 2] - in simple terms

        💡 What to know:
        • [Key fact 1]
        • [Key fact 2]

        🏥 Next steps:
        • [Action 1]
        • [Action 2]

        For disease queries (disease_info):
        📖 About [Disease Name]:

        🔍 What it is:
        • [Simple explanation]

        📋 Common signs:
        • [Symptom 1]
        • [Symptom 2]

        💊 Management:
        • [Treatment 1]
        • [Treatment 2]

        🛡️ Prevention:
        • [Prevention tip 1]

        FINAL REQUIREMENTS:
        - MAX 10 bullet points total
        - Simple language a 12-year-old can understand
        - Use emojis to make it friendly
        - Skip complex medical jargon
        - Include "Consult a doctor for proper care" at the end
        """

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """
**User Question:** {query}
**Condition:** {disease_name}
**Medical Info:** {raw_result}
**Query Type:** {decision}

Please create a short, easy-to-understand response:
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
        decision: str = "",  # Add this parameter
    ) -> str:
        """Format medical information into short, simple points"""
        response = self.chain.invoke(
            {
                "query": query,
                "disease_name": disease_name,
                "raw_result": raw_result,
                "decision": decision,  # Include decision in the prompt
            }
        )
        return response.content.strip()

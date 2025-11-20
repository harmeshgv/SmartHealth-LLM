import logging
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class FormatterAgent(BaseAgent):
    name: str = "Formatter Agent"
    description: str = "Formats complex medical information into a short, simple, easy-to-understand, point-wise response for the user, considering the conversation history."

    def __init__(self, llm: Any):
        super().__init__(llm_instance=llm)
        logger.info("FormatterAgent initialized.")

        self.system_prompt = """
        You are a friendly and helpful medical response formatter. Your task is to transform complex, raw medical information into a short, easy-to-understand, point-wise response. You should consider the entire conversation to make your response feel natural and relevant.

        RULES:
        1. KEEP IT SHORT & SIMPLE - Maximum 8-10 key points total.
        2. USE EMOJIS for visual appeal and clarity (e.g., 🔍, 📋, 💡, 🏥).
        3. USE PLAIN LANGUAGE - Explain complex medical terms in simple, everyday words.
        4. POINT-WISE FORMAT - Use bullet points (•) for readability. No long paragraphs.
        5. FOCUS ON KEY INFORMATION - Prioritize the most important and actionable details.
        6. FRIENDLY & REASSURING TONE - Be empathetic and supportive.
        7. ALWAYS end with the exact line: "⚠️ Disclaimer: This is for informational purposes only. Please consult a doctor for proper medical advice."

        FORMATTING GUIDELINES (adapt based on the context):

        If the user asked about symptoms (decision type is "symptom_to_disease"):
        🔍 Based on your symptoms, one possibility could be: **[Disease Name]**

        📋 Main Symptoms Can Include:
        • [Symptom 1] - in simple terms
        • [Symptom 2] - in simple terms

        💡 What to Know:
        • [Key fact 1 about the condition]
        • [Key fact 2 about the condition]

        🏥 Recommended Next Steps:
        • [Action 1, e.g., "Consider speaking with a healthcare professional."]
        • [Action 2, e.g., "Rest and stay hydrated."]

        If the user asked for disease information (decision type is "disease_info"):
        📖 Here's a little about **[Disease Name]**:

        🔍 What It Is:
        • [Simple, one-sentence explanation of the disease]

        📋 Common Signs:
        • [Symptom 1]
        • [Symptom 2]

        💊 Typical Management:
        • [Treatment/management option 1]
        • [Treatment/management option 2]
        
        If the query was unclear (decision type is "biomedical_ner"):
        ℹ️ It seems you're asking about a few things. Here's a quick summary:
        
        • **[Entity 1]**: [Brief, simple explanation]
        • **[Entity 2]**: [Brief, simple explanation]
        • For more specific details, could you ask a more focused question about one of these?

        If an image was analyzed (decision type is "image_analysis"):
        🖼️ Based on the image analysis, one possibility is **[Disease Name]**.

        🔍 What It Is:
        • [Simple explanation of the predicted condition]

        📋 Common Signs Often Include:
        • [Symptom 1]
        • [Symptom 2]

        🏥 Recommended Next Steps:
        • Monitor the area for any changes in size, color, or shape.
        • It's always a good idea to have a doctor take a look for an accurate diagnosis.
        """

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    """
**My Original Question:** {query}
**Identified Condition:** {disease_name}
**Available Medical Info:** {raw_result}
**Query Type:** {decision}

Please create a short, easy-to-understand response based on these details and our conversation.
""",
                ),
            ]
        )
        self.chain = self.prompt_template | self.llm

    async def invoke(self, state: Dict[str, Any], **kwargs: Any) -> str:
        """Formats medical information into a short, simple, point-wise response."""
        logger.debug(f"FormatterAgent invoked with state: {state}")
        
        # Ensure all necessary fields are present, providing defaults if not
        current_state = {
            "query": state.get("query", "N/A"),
            "disease_name": state.get("disease_name", "Not specified"),
            "raw_result": state.get("raw_result", "No information available."),
            "decision": state.get("decision", "N/A"),
            "chat_history": state.get("chat_history", []),
            **kwargs,
        }
        
        response = await self.chain.ainvoke(current_state)
        
        logger.info("FormatterAgent successfully formatted the response.")
        return response.content.strip()

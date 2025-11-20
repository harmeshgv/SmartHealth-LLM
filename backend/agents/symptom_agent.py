import logging
from typing import TypedDict, Optional, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

from .base_agent import BaseAgent
from ..tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# This state is internal to the SymptomToDiseaseAgent's own graph
class InternalSymptomState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    symptoms: Optional[list]
    ner_entities: Optional[list]
    matched_diseases: Optional[list]
    final_disease: Optional[str]

class SymptomToDiseaseAgent(BaseAgent):
    name: str = "Symptom to Disease Agent"
    description: str = "Extracts symptoms from a user query in the context of the conversation, matches them to potential diseases, and decides on the most likely one."

    def __init__(self, llm: Any, tool_registry: ToolRegistry):
        super().__init__(llm_instance=llm)
        self.tool_registry = tool_registry
        self.graph = self._build_graph()
        logger.info("SymptomToDiseaseAgent initialized.")

    # --- Internal Prompts ---
    def _build_symptom_extraction_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical symptom extractor. Your task is to extract ALL symptoms mentioned in the user's *most recent* message, using the conversation history for context.
                    CRITICAL: You MUST return symptoms even if they are described in everyday language.
                    Examples: "I have fever and headache" → "fever, headache"; "My throat hurts and I'm coughing" → "sore throat, cough".
                    Return ONLY a comma-separated list of symptoms based on the last user message. If no symptoms are mentioned in the last message, return "none".
                    """,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_query}"), # This will be the last user message
            ]
        )

    def _build_disease_decision_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical assistant. Your ONLY task is to identify the SINGLE most likely disease based on the given symptoms and matches.
                    Available Disease Matches: {matched_diseases}
                    User Symptoms: {symptoms}
                    CRITICAL RULES: Return ONLY the disease name as plain text. NO explanations. If unsure, return "unknown".
                    Disease:""",
                )
            ]
        )

    # --- Internal Graph Nodes ---
    async def _extract_entities(self, state: InternalSymptomState):
        logger.debug("SymptomAgent: Extracting entities from query.")
        ner_tool = self.tool_registry.get_tool("biomedical_ner_tool")
        ner_result = await ner_tool.execute(text=state["user_query"])
        return {**state, "ner_entities": ner_result.get("entities", [])}

    async def _extract_symptoms(self, state: InternalSymptomState):
        logger.debug("SymptomAgent: Extracting symptoms.")
        ner_entities = state.get("ner_entities", [])

        symptoms_from_ner = [entity['word'] for entity in ner_entities if entity.get("entity_group") == "symptom"]

        if symptoms_from_ner:
            logger.info(f"SymptomAgent: Extracted symptoms from NER: {symptoms_from_ner}")
            return {**state, "symptoms": symptoms_from_ner}

        logger.info("SymptomAgent: No symptoms found via NER, falling back to LLM extraction.")
        prompt = self._build_symptom_extraction_prompt()
        chain = prompt | self.llm
        
        response = await chain.ainvoke({
            "chat_history": state.get("chat_history", []),
            "user_query": state.get("user_query", "")
        })
        symptoms_str = response.content.strip()

        if symptoms_str.lower() == "none":
            return {**state, "symptoms": []}

        symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
        logger.info(f"SymptomAgent: LLM extracted symptoms: {symptoms}")
        return {**state, "symptoms": symptoms}

    async def _match_symptom_to_disease(self, state: InternalSymptomState):
        logger.debug("SymptomAgent: Matching symptoms to diseases.")
        symptoms = state.get("symptoms", [])
        if not symptoms:
            return {**state, "matched_diseases": [], "final_disease": "unknown"}

        matcher_tool = self.tool_registry.get_tool("symptom_disease_matcher_tool")
        matches_result = await matcher_tool.execute(symptoms=symptoms)
        logger.debug(f"SymptomAgent: Disease matcher result: {matches_result}")
        return {**state, "matched_diseases": matches_result.get("matched_diseases", [])}

    async def _decide_final_disease(self, state: InternalSymptomState):
        logger.debug("SymptomAgent: Deciding on final disease from matches.")
        matched_diseases = state.get("matched_diseases", [])
        if not matched_diseases:
            return {**state, "final_disease": "unknown"}

        prompt = self._build_disease_decision_prompt()
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "matched_diseases": str(matched_diseases),
            "symptoms": ", ".join(state.get("symptoms", [])),
        })
        disease = response.content.strip().replace('"', "").replace("'", "").strip()
        logger.info(f"SymptomAgent: Final disease decision: {disease}")
        return {**state, "final_disease": disease}

    # --- Graph Builder ---
    def _build_graph(self):
        graph = StateGraph(InternalSymptomState)

        graph.add_node("ner", self._extract_entities)
        graph.add_node("extract_symptoms", self._extract_symptoms)
        graph.add_node("map_disease", self._match_symptom_to_disease)
        graph.add_node("decide_disease", self._decide_final_disease)

        graph.set_entry_point("ner")
        graph.add_edge("ner", "extract_symptoms")
        graph.add_edge("extract_symptoms", "map_disease")
        graph.add_edge("map_disease", "decide_disease")
        graph.add_edge("decide_disease", END)

        return graph.compile()

    # --- Main Invoke Method ---
    async def invoke(self, state: dict, **kwargs: Any) -> dict:
        """
        The main entry point for the SymptomToDiseaseAgent.
        It runs its internal graph to determine a likely disease from symptoms.
        """
        user_query = state.get("query")
        chat_history = state.get("chat_history", [])
        logger.info(f"SymptomToDiseaseAgent invoked for query: '{user_query}'.")

        internal_state = {
            "user_query": user_query,
            "chat_history": chat_history,
        }
        final_internal_state = await self.graph.ainvoke(internal_state)

        # Return the final disease name to be merged into the main orchestration state
        return {"disease_name": final_internal_state.get("final_disease", "unknown")}


if __name__ == "__main__":
    import asyncio

    # Import LLM setup
    from backend.utils.llm import set_llm
    from backend.config import settings

    # Import ToolRegistry (auto-loads all tools)
    from backend.tool_registry import ToolRegistry

    # Initialize LLM
    llm = set_llm(
        api_key=settings.TEST_API_KEY,
        api_key_base=settings.TEST_API_BASE,
        model=settings.TEST_MODEL
    )

    # Initialize registry (this auto-loads all tools)
    tool_registry = ToolRegistry()

    # Initialize agent
    agent = SymptomToDiseaseAgent(llm, tool_registry)

    # Test state
    state = {"query": "I am having cough and dry throat for a week", "disease_name": ""}

    result = asyncio.run(agent.invoke(state))
    print("\n===== FINAL RESULT =====")
    print(result)
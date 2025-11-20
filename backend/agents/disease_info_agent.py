import logging
from typing import TypedDict, Optional, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

from .base_agent import BaseAgent
from ..tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# This state is internal to the DiseaseInfoAgent's own graph
class InternalDiseaseState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage]
    ner_entities: Optional[list]
    disease_name: str
    db_info: Optional[dict]
    web_info: Optional[str]
    final_info: str
    decision: str

class DiseaseInfoAgent(BaseAgent):
    name: str = "Disease Information Agent"
    description: str = "An agent that identifies a specific disease from a query and conversation history, then gathers detailed information about it."

    def __init__(self, llm: Any, tool_registry: ToolRegistry):
        super().__init__(llm_instance=llm)
        self.tool_registry = tool_registry
        self.graph = self._build_graph()
        logger.info("DiseaseInfoAgent initialized with multi-step logic.")

    # --- Internal Prompts ---
    def _build_disease_extraction_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a medical disease identifier. Your task is to analyze the user's latest query in the context of the conversation history and identify the single disease or condition they are asking about.

Rules:
- Prioritize the disease explicitly mentioned in the most recent user query.
- Use the conversation history to resolve ambiguity. For example, if the user asks "what about that?", use the history to find what "that" refers to.
- If any NER entity is a clear disease/condition, use that.
- Return ONLY the disease name as a clean lowercase string.
- If you truly cannot infer a disease from the latest query and history, return "unknown".

Inputs:
NER Entities: {entity_context}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_query}"),
            ("system", "Based on the latest query and conversation history, what is the single disease being discussed? Return only the name.")
        ])


    def _build_synth_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a router. Your only job is to decide if the provided `Database Information` is sufficient to answer the `User Query`.
Do not answer the user's query. Do not be helpful.
Your output MUST be a single word: `search` or `ok`.

- `search`: The database information is insufficient or missing.
- `ok`: The database information is sufficient.

User Query: {user_query}
Conversation History: {chat_history}

Database Information:
{db_info}
"""),
            ("system", "Decision (search/ok):")
        ])

    # --- Internal Graph Nodes ---
    async def _extract_entities(self, state: InternalDiseaseState):
        logger.debug(f"DiseaseInfoAgent: (Step 1) Extracting entities from query: '{state['user_query']}'")
        ner_tool = self.tool_registry.get_tool("biomedical_ner_tool")
        ner_result = await ner_tool.execute(text=state["user_query"])
        return {**state, "ner_entities": ner_result.get("entities", [])}

    async def _extract_disease_name(self, state: InternalDiseaseState):
        logger.debug(f"DiseaseInfoAgent: (Step 2) Extracting disease name using LLM.")
        prompt = self._build_disease_extraction_prompt()
        chain = prompt | self.llm
        
        response = await chain.ainvoke({
            "entity_context": state.get("ner_entities", []),
            "chat_history": state.get("chat_history", []),
            "user_query": state.get("user_query", "")
        })

        disease_name = response.content.strip().lower().replace('"', "").replace("'", "")
        logger.info(f"DiseaseInfoAgent: LLM extracted disease name: '{disease_name}'")
        return {**state, "disease_name": disease_name}

    async def _retrieve_from_db(self, state: InternalDiseaseState):
        logger.debug(f"DiseaseInfoAgent: (Step 3) Retrieving '{state['disease_name']}' from DB.")
        retriever_tool = self.tool_registry.get_tool("disease_info_retriever_tool")
        result = await retriever_tool.execute(disease_name=state['disease_name'])
        return {**state, "db_info": result.get("info")}

    async def _make_decision(self, state: InternalDiseaseState):
        logger.debug("DiseaseInfoAgent: (Step 4) Deciding if web search is needed.")
        db_info = state.get("db_info")
        if not db_info or (isinstance(db_info, dict) and "error" in db_info):
            logger.info("DiseaseInfoAgent: DB info missing or contains an error. Deciding to search.")
            return {**state, "decision": "search"}

        prompt = self._build_synth_prompt()
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "db_info": db_info,
            "chat_history": state.get("chat_history", []),
            "user_query": state.get("user_query", "")
        })
        decision = response.content.strip().lower()
        logger.info(f"DiseaseInfoAgent: LLM decision for web search is '{decision}'.")
        return {**state, "decision": decision}

    async def _perform_search(self, state: InternalDiseaseState):
        logger.debug(f"DiseaseInfoAgent: (Step 5) Performing web search for '{state['disease_name']}'.")
        search_tool = self.tool_registry.get_tool("google_search_tool")
        result = await search_tool.execute(query=f"medical information about {state['disease_name']}")
        return {**state, "web_info": result.get("result")}

    async def _synthesize_info(self, state: InternalDiseaseState):
        logger.debug("DiseaseInfoAgent: (Step 6) Synthesizing final information.")
        db_info_str = f"Database Info: {state.get('db_info', 'Not found.')}"
        web_info_str = f"Web Search Info: {state.get('web_info', 'Not found.')}"
        final_info = f"{db_info_str}\n\n{web_info_str}"
        return {**state, "final_info": final_info}

    # --- Graph Builder ---
    def _build_graph(self):
        graph = StateGraph(InternalDiseaseState)

        graph.add_node("ner", self._extract_entities)
        graph.add_node("extract_disease", self._extract_disease_name)
        graph.add_node("db_lookup", self._retrieve_from_db)
        graph.add_node("decision", self._make_decision)
        graph.add_node("search", self._perform_search)
        graph.add_node("synthesize", self._synthesize_info)

        graph.set_entry_point("ner")
        graph.add_edge("ner", "extract_disease")
        graph.add_edge("extract_disease", "db_lookup")
        graph.add_edge("db_lookup", "decision")
        graph.add_conditional_edges(
            "decision",
            lambda state: state.get("decision", "search"),
            {"ok": "synthesize", "search": "search"}, # Go to synthesize even if ok
        )
        graph.add_edge("search", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    # --- Main Invoke Method ---
    async def invoke(self, state: dict, **kwargs: Any) -> dict:
        """
        The main entry point for the DiseaseInfoAgent.
        It runs its internal graph to gather information about a disease.
        """
        user_query = state.get("query")
        chat_history = state.get("chat_history", [])
        disease_name = state.get("disease_name", "") # Pre-filled from orchestrator if available

        logger.info(f"DiseaseInfoAgent invoked. Query: '{user_query}', Pre-filled Disease: '{disease_name}'")

        internal_state = {
            "user_query": user_query,
            "chat_history": chat_history,
            "disease_name": disease_name,
        }

        final_internal_state = await self.graph.ainvoke(internal_state)

        # Return the gathered information to be merged into the main orchestration state
        return {"raw_result": final_internal_state.get("final_info", final_internal_state.get("db_info"))}


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
    agent = DiseaseInfoAgent(llm, tool_registry)

    # Test state
    state = {"query": "What is the symptoms of asthma?", "disease_name": ""}

    result = asyncio.run(agent.invoke(state))
    print("\n===== FINAL RESULT =====")
    print(result)

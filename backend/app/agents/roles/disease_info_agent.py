import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from app.agents.base.base_agent import BaseAgent
from app.tools.medical.biomedical_ner_tool import BiomedicalNERTool
from app.tools.medical.disease_info_tool import DiseaseInfoRetrieverTool
from app.tools.internet.google_search_tool import GoogleSearchTool
from app.utils.prompt_loader import load_prompt


# ---------------------------
# STATE STRUCTURE
# ---------------------------
class DiseaseInfoState(TypedDict):
    user_message: str
    ner_entities: Dict[str, Any]  # {symptoms:[], diseases:[], ...}
    llm_output: str
    extracted_disease: str
    disease_info: str
    db_success: bool


# ---------------------------
# AGENT
# ---------------------------
class DiseaseInfoAgent(BaseAgent):
    agent_name = "disease_info_agent"   # MUST match decider output

    def __init__(self, context):
        super().__init__(context)
        # Instantiate tools ONCE
        self.ner_tool = BiomedicalNERTool()
        self.db_tool = DiseaseInfoRetrieverTool()
        self.google = GoogleSearchTool()


    # ---------------------------
    async def ner_node(self, state: DiseaseInfoState):
        entities = await self.ner_tool.run(state["user_message"])
        state["ner_entities"] = entities
        return state


    # ---------------------------
    async def llm_node(self, state: DiseaseInfoState):
        template = load_prompt("disease_info_prompt.txt")

        prompt = (
            template
            .replace("{{INPUT}}", state["user_message"])
            .replace("{{NER_CONTEXT}}", json.dumps(state["ner_entities"]))
        )

        raw = await self.llm_call(prompt)
        state["llm_output"] = raw.strip()
        return state



    # ---------------------------

    # ---------------------------
    async def parse_node(self, state):
        # HARD OVERRIDE: if NER found disease, use it
        diseases = state["ner_entities"].get("diseases", [])
        if diseases:
            state["extracted_disease"] = diseases[0]  # take top one
            return state

        # Otherwise use LLM JSON
        raw = state["llm_output"]
        try:
            parsed = json.loads(raw)
            state["extracted_disease"] = parsed.get("disease", "none")
        except:
            state["extracted_disease"] = "none"

        return state



    # ---------------------------
    async def local_db_node(self, state: DiseaseInfoState):
        disease = state["extracted_disease"]
        retrieved = await self.db_tool.run(disease)

        if retrieved:
            state["disease_info"] = retrieved
            state["db_success"] = True
        else:
            state["db_success"] = False

        return state


    # ---------------------------
    async def web_search_node(self, state: DiseaseInfoState):
        search = await self.google.run(state["extracted_disease"])
        state["disease_info"] = search
        return state


    async def memory_node(self, state: DiseaseInfoState):
        await self.save_memory(
            user_message=state["user_message"],
            llm_output=state["llm_output"]
        )
        return state

    # ---------------------------
    # BUILD GRAPH
    # ---------------------------
    def build_graph(self):
        workflow = StateGraph(DiseaseInfoState)

        workflow.add_node("ner", self.ner_node)
        workflow.add_node("llm", self.llm_node)
        workflow.add_node("parse", self.parse_node)
        workflow.add_node("local_db", self.local_db_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("memory", self.memory_node)

        workflow.set_entry_point("ner")

        workflow.add_edge("ner", "llm")
        workflow.add_edge("llm", "parse")
        workflow.add_edge("parse", "local_db")

        # CONDITIONAL FALLBACK
        workflow.add_conditional_edges(
            "local_db",
            lambda s: "memory" if s["db_success"] else "web_search",
            {
                "memory": "memory",
                "web_search": "web_search"
            }
        )

        workflow.add_edge("memory", END)
        workflow.add_edge("web_search", END)

        return workflow.compile()

    # ---------------------------
    # RUN
    # ---------------------------
    async def run(self, user_message: str, **kwargs):
        """
        Supports two modes:
        MODE 1 → user asked about a disease → extract using NER + LLM
        MODE 2 → symptom matcher already provided disease_matched
        """

        graph = self.build_graph()

        # MODE 2: disease already matched by symptom agent
        if "disease_matched" in kwargs and kwargs["disease_matched"]:

            disease_raw = kwargs["disease_matched"]

            # FIX HERE → Extract actual string
            if isinstance(disease_raw, dict):
                disease = disease_raw.get("disease") or disease_raw.get("name")
            else:
                disease = disease_raw

            # Now safe to use
            db_data = await self.db_tool.run(disease)

            if db_data:
                return {
                    "disease": disease,
                    "info": db_data
                }

            google = await self.google.run(disease)
            return {
                "disease": disease,
                "info": google
            }


        # MODE 1: normal query → extract disease from user_message
        result = await graph.ainvoke({
            "user_message": user_message,
            "ner_entities": {},
            "llm_output": "",
            "extracted_disease": "",
            "disease_info": "",
            "db_success": False
        })

        return {
            "disease": result["extracted_disease"],
            "info": result["disease_info"]
        }


# Manual debug runner
if __name__ == "__main__":
    import asyncio
    from app.core.agent_context import AgentContext

    async def test():
        context = AgentContext()
        context.debug = True

        agent = DiseaseInfoAgent(context)
        result = await agent.run("what is dengue")
        print("\nFinal Output:", result)

    asyncio.run(test())
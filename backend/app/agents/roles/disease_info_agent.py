import logging
import json
import time
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from app.agents.base.base_agent import BaseAgent
from app.tools.medical.biomedical_ner_tool import BiomedicalNERTool
from app.tools.medical.disease_info_tool import DiseaseInfoRetrieverTool
from app.tools.internet.google_search_tool import GoogleSearchTool
from app.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class DiseaseInfoState(TypedDict):
    user_message: str
    ner_entities: Dict[str, Any]
    llm_output: str
    extracted_disease: str
    disease_info: str
    db_success: bool
    run_id: str


class DiseaseInfoAgent(BaseAgent):
    agent_name = "disease_info_agent"

    def __init__(self, context):
        super().__init__(context)
        if not hasattr(self, "tools") or self.tools is None:
            self.tools = {}

        if "biomedical_ner_tool" not in self.tools:
            self.tools["biomedical_ner_tool"] = BiomedicalNERTool()
        if "disease_info_retriever_tool" not in self.tools:
            self.tools["disease_info_retriever_tool"] = DiseaseInfoRetrieverTool()

        self.ner_tool = self.tools["biomedical_ner_tool"]
        self.db_tool = self.tools["disease_info_retriever_tool"]

        if "google_search_tool" not in self.tools:
            try:
                self.tools["google_search_tool"] = GoogleSearchTool()
            except Exception:
                logger.warning(
                    "Google search tool unavailable; continuing with local DB only.",
                    extra={"agent": self.agent_name},
                )
                self.tools["google_search_tool"] = None

        self.google = self.tools["google_search_tool"]

    async def ner_node(self, state: DiseaseInfoState):
        run_id = state["run_id"]
        logger.info("Node: ner", extra={"run_id": run_id, "agent": self.agent_name})
        entities = await self.ner_tool.run(state["user_message"], run_id=run_id)
        state["ner_entities"] = entities
        return state

    async def llm_node(self, state: DiseaseInfoState):
        run_id = state["run_id"]
        logger.info("Node: llm", extra={"run_id": run_id, "agent": self.agent_name})
        template = load_prompt("disease_info_prompt.txt")

        prompt = template.replace("{{INPUT}}", state["user_message"]).replace(
            "{{NER_CONTEXT}}", json.dumps(state["ner_entities"])
        )

        raw = await self.llm_call(prompt, run_id=run_id)
        state["llm_output"] = raw.strip()
        return state

    async def parse_node(self, state):
        run_id = state["run_id"]
        logger.info("Node: parse", extra={"run_id": run_id, "agent": self.agent_name})
        diseases = state["ner_entities"].get("diseases", [])
        if diseases:
            state["extracted_disease"] = diseases[0]
            return state

        raw = state["llm_output"]
        try:
            parsed = json.loads(raw)
            state["extracted_disease"] = parsed.get("disease", "none")
        except:
            state["extracted_disease"] = "none"

        return state

    async def local_db_node(self, state: DiseaseInfoState):
        run_id = state["run_id"]
        logger.info("Node: local_db", extra={"run_id": run_id, "agent": self.agent_name})
        disease = state.get("extracted_disease")

        if not disease:
            state["db_success"] = False
            return state

        try:
            retrieved = await self.db_tool.run(disease, run_id=run_id)

            if retrieved and isinstance(retrieved, dict) and "error" not in retrieved:
                state["disease_info"] = retrieved
                state["db_success"] = True
            else:
                state["db_success"] = False

        except Exception as e:
            logger.error(f"Error while querying DB for disease: {disease}", extra={"run_id": run_id, "agent": self.agent_name}, exc_info=True)
            state["db_success"] = False

        return state

    async def web_search_node(self, state: DiseaseInfoState):
        run_id = state["run_id"]
        logger.info("Node: web_search", extra={"run_id": run_id, "agent": self.agent_name})
        if not self.google:
            state["disease_info"] = {
                "error": "Web search unavailable. Configure SERPER_API_KEY to enable live search."
            }
            return state

        search = await self.google.run(state["extracted_disease"], run_id=run_id)
        state["disease_info"] = search
        return state

    async def memory_node(self, state: DiseaseInfoState):
        run_id = state["run_id"]
        logger.info("Node: memory", extra={"run_id": run_id, "agent": self.agent_name})
        await self.save_memory(
            user_message=state["user_message"], llm_output=state["llm_output"], run_id=run_id
        )
        return state

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

        workflow.add_conditional_edges(
            "local_db",
            lambda s: "memory" if s["db_success"] else "web_search",
            {"memory": "memory", "web_search": "web_search"},
        )

        workflow.add_edge("web_search", "memory")
        workflow.add_edge("memory", END)

        return workflow.compile()

    async def run(self, user_message: str, run_id: str, **kwargs):
        start_time = time.time()
        logger.info("Agent run started", extra={"run_id": run_id, "agent": self.agent_name})
        
        graph = self.build_graph()

        if "disease_matched" in kwargs and kwargs["disease_matched"]:
            disease_raw = kwargs["disease_matched"]
            if isinstance(disease_raw, dict):
                disease = disease_raw.get("disease") or disease_raw.get("name")
            else:
                disease = disease_raw

            db_data = await self.db_tool.run(disease, run_id=run_id)

            if db_data and isinstance(db_data, dict) and "error" not in db_data:
                info = db_data
            else:
                if self.google:
                    info = await self.google.run(disease, run_id=run_id)
                else:
                    info = {
                        "error": "No local match found and web search is unavailable."
                    }

            end_time = time.time()
            latency = end_time - start_time
            logger.info("Agent run finished (direct path)", extra={"run_id": run_id, "agent": self.agent_name, "latency": latency})
            
            return {"disease": disease, "info": info}

        result = await graph.ainvoke(
            {
                "user_message": user_message,
                "ner_entities": {},
                "llm_output": "",
                "extracted_disease": "",
                "disease_info": "",
                "db_success": False,
                "run_id": run_id
            }
        )

        end_time = time.time()
        latency = end_time - start_time
        logger.info("Agent run finished (graph path)", extra={"run_id": run_id, "agent": self.agent_name, "latency": latency})

        return {"disease": result["extracted_disease"], "info": result["disease_info"]}

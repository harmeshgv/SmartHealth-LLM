import logging
import json
import time
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from app.agents.base.base_agent import BaseAgent
from app.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class ReasoningState(TypedDict):
    user_message: str
    symptoms: list
    disease_matched: str
    disease_info: Dict[str, Any]
    final_output: str
    run_id: str


class ReasoningAgent(BaseAgent):
    agent_name = "reasoning_agent"

    async def llm_node(self, state: ReasoningState):
        run_id = state["run_id"]
        logger.info("Node: llm", extra={"run_id": run_id, "agent": self.agent_name})

        template = load_prompt("reasoning_prompt.txt")

        prompt = (
            template.replace("{{USER_MESSAGE}}", state["user_message"])
            .replace("{{SYMPTOMS}}", json.dumps(state.get("symptoms", [])))
            .replace("{{DISEASE}}", state.get("disease_matched", "unknown"))
            .replace("{{DISEASE_INFO}}", json.dumps(state.get("disease_info", {})))
        )

        raw = await self.llm_call(prompt, run_id=run_id)

        state["final_output"] = raw.strip()
        return state

    def build_graph(self):
        workflow = StateGraph(ReasoningState)
        workflow.add_node("llm", self.llm_node)
        workflow.set_entry_point("llm")
        workflow.add_edge("llm", END)
        return workflow.compile()

    async def run(self, user_message: str, run_id: str, **kwargs):
        start_time = time.time()
        logger.info("Agent run started", extra={"run_id": run_id, "agent": self.agent_name})

        graph = self.build_graph()

        result = await graph.ainvoke(
            {
                "user_message": user_message,
                "symptoms": kwargs.get("symptoms", []),
                "disease_matched": kwargs.get("disease_matched", "unknown"),
                "disease_info": kwargs.get("info", {}),
                "final_output": "",
                "run_id": run_id,
            }
        )

        end_time = time.time()
        latency = end_time - start_time
        logger.info("Agent run finished", extra={"run_id": run_id, "agent": self.agent_name, "latency": latency})

        return {"final_output": result["final_output"]}

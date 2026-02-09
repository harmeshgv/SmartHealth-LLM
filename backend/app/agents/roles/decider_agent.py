import json
import logging
import time
import re
from app.agents.base.base_agent import BaseAgent
from langgraph.graph import StateGraph, END
from app.utils.prompt_loader import load_prompt
from typing import List, TypedDict

logger = logging.getLogger(__name__)


class DeciderState(TypedDict):
    user_message: str
    llm_output: str
    intent: str
    agents: List[str]
    run_id: str


class DeciderAgent(BaseAgent):
    agent_name = "decider_agent"
    _ALLOWED_AGENTS = {
        "conversation_agent",
        "symptom_matcher_agent",
        "disease_info_agent",
    }
    _AGENT_ALIASES = {
        "converstion_agent": "conversation_agent",
        "symptom_agent": "symptom_matcher_agent",
        "disease_agent": "disease_info_agent",
        "info_agent": "disease_info_agent",
    }

    def _heuristic_route(self, user_message: str) -> dict:
        text = (user_message or "").strip().lower()

        casual_patterns = [
            r"^(hi|hello|hey|yo|good morning|good afternoon|good evening)\b",
            r"\b(how are you|what's up|thank you|thanks)\b",
        ]
        symptom_keywords = {
            "fever", "cough", "cold", "pain", "headache", "nausea",
            "vomit", "vomiting", "sore throat", "fatigue", "rash",
            "dizziness", "diarrhea", "chest pain", "breath",
        }
        disease_info_keywords = {
            "what is", "tell me about", "explain", "information", "about",
            "cause", "treatment", "prevent", "symptoms of", "disease",
        }

        if any(re.search(pattern, text) for pattern in casual_patterns):
            return {"intent": "conversation", "agents": ["conversation_agent"]}

        if any(keyword in text for keyword in symptom_keywords):
            return {
                "intent": "symptom_analysis",
                "agents": ["symptom_matcher_agent", "disease_info_agent"],
            }

        if any(keyword in text for keyword in disease_info_keywords):
            return {"intent": "disease_information", "agents": ["disease_info_agent"]}

        return {"intent": "conversation", "agents": ["conversation_agent"]}

    def _sanitize_agents(self, agents: list) -> list:
        cleaned = []
        for agent in agents:
            if not isinstance(agent, str):
                continue
            normalized = self._AGENT_ALIASES.get(agent.strip(), agent.strip())
            if normalized in self._ALLOWED_AGENTS and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    async def llm_node(self, state: dict):
        run_id = state["run_id"]
        logger.info("Node: llm", extra={"run_id": run_id, "agent": self.agent_name})

        template = load_prompt("decider_prompt.txt")
        prompt = template.replace("{{INPUT}}", state["user_message"])

        raw = await self.llm_call(prompt, run_id=run_id)

        logger.info("LLM raw output", extra={"run_id": run_id, "agent": self.agent_name, "output": raw})
        state["llm_output"] = raw.strip()
        return state

    async def parse_node(self, state: dict):
        run_id = state["run_id"]
        logger.info("Node: parse", extra={"run_id": run_id, "agent": self.agent_name})
        raw = state.get("llm_output", "")
        heuristic_plan = self._heuristic_route(state.get("user_message", ""))
        parse_failed = False

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parse_failed = True
            logger.warning(
                "Failed to parse LLM output, using fallback", extra={"run_id": run_id, "agent": self.agent_name}
            )
            parsed = {}

        parsed_agents = parsed.get("agents", [])
        if isinstance(parsed_agents, str):
            parsed_agents = [parsed_agents]

        sanitized_agents = self._sanitize_agents(parsed_agents)
        if not sanitized_agents:
            sanitized_agents = heuristic_plan["agents"]

        state["intent"] = "fallback" if parse_failed else (parsed.get("intent") or heuristic_plan["intent"])
        state["agents"] = sanitized_agents
        return state

    def build_graph(self):
        workflow = StateGraph(DeciderState)

        async def llm_wrapper(state):
            return await self.llm_node(state)

        async def parse_wrapper(state):
            return await self.parse_node(state)

        workflow.add_node("llm", llm_wrapper)
        workflow.add_node("parse", parse_wrapper)
        workflow.set_entry_point("llm")
        workflow.add_edge("llm", "parse")
        workflow.add_edge("parse", END)

        graph = workflow.compile()
        return graph

    async def run(self, user_message: str, run_id: str):
        start_time = time.time()
        logger.info("Agent run started", extra={"run_id": run_id, "agent": self.agent_name})

        try:
            graph = self.build_graph()

            result = await graph.ainvoke(
                {
                    "user_message": user_message,
                    "llm_output": "",
                    "intent": "",
                    "agents": [],
                    "run_id": run_id,
                }
            )

            end_time = time.time()
            latency = end_time - start_time

            logger.info(
                "Agent run finished",
                extra={
                    "run_id": run_id,
                    "agent": self.agent_name,
                    "intent": result["intent"],
                    "agents": result["agents"],
                    "latency": latency,
                },
            )

            return {
                "intent": result["intent"],
                "agents": result["agents"],
            }

        except Exception:
            logger.error(
                "Agent run failed",
                extra={"run_id": run_id, "agent": self.agent_name},
                exc_info=True,
            )
            raise

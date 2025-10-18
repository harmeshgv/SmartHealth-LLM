from langgraph.graph import StateGraph, END
from agents.decider_agent import DeciderAgent
from agents.disease_info_agent import DiseaseInfoAgent
from agents.symptom_agent import SymptomToDiseaseAgent
from agents.formatter_agent import FormatterAgent
from utils.llm import set_llm
from dotenv import load_dotenv
from typing import TypedDict


class AgentState(TypedDict):
    query: str
    decision: str  # "disease_info" or "symptom_match"
    disease_name: str
    raw_result: str  # Raw result from disease info agent
    formatted_result: str  # Final formatted result


class AgentOrchestration:
    def __init__(self, llm_instance):

        self.llm_instance = llm_instance

        # Instantiate all agents including formatter
        self.DECIDER_AGENT = DeciderAgent(self.llm_instance)
        self.SYMPTOM_AGENT = SymptomToDiseaseAgent(self.llm_instance)
        self.DISEASE_INFO_AGENT = DiseaseInfoAgent(self.llm_instance)
        self.FORMATTER_AGENT = FormatterAgent(self.llm_instance)

        # Initialize workflow
        self.workflow = StateGraph(AgentState)

        # Add all nodes including formatter
        self.workflow.add_node("decider", self.decider_node)
        self.workflow.add_node("symptom_to_disease", self.symptom_node)
        self.workflow.add_node("disease_info", self.disease_info_node)
        self.workflow.add_node("formatter", self.formatter_node)

        # Conditional routing from decider
        def route_decision(state: AgentState) -> str:
            decision = state.get("decision", "").lower()
            if "disease_info" in decision:
                return "disease_info"
            elif "symptom" in decision:
                return "symptom_to_disease"
            return "disease_info"

        # Add conditional edge from decider
        self.workflow.add_conditional_edges(
            "decider",
            route_decision,
            {
                "disease_info": "disease_info",
                "symptom_to_disease": "symptom_to_disease",
            },
        )

        # Symptom → disease info
        self.workflow.add_edge("symptom_to_disease", "disease_info")

        # Disease info → formatter (ALWAYS goes to formatter after disease info)
        self.workflow.add_edge("disease_info", "formatter")

        # Formatter ends workflow
        self.workflow.add_edge("formatter", END)

        # Set entrypoint
        self.workflow.set_entry_point("decider")

        # Compile workflow once
        self.app = self.workflow.compile()

    def decider_node(self, state: AgentState) -> AgentState:
        print(f"🔍 Decider Node - Query: {state['query']}")
        decision = self.DECIDER_AGENT.invoke(state["query"])
        print(f"🔍 Decider Decision: {decision}")
        return {
            "decision": decision,
            "query": state["query"],
            "disease_name": state.get("disease_name", ""),
            "raw_result": state.get("raw_result", ""),
        }

    def symptom_node(self, state: AgentState) -> AgentState:
        print(f"🔍 Symptom Node - Query: {state['query']}")
        disease_name = self.SYMPTOM_AGENT.invoke(state["query"])
        print(f"🔍 Symptom Agent Output: {disease_name}")
        return {
            "disease_name": disease_name,
            "query": state["query"],
            "decision": state.get("decision", ""),
            "raw_result": state.get("raw_result", ""),
        }

    def disease_info_node(self, state: AgentState) -> AgentState:
        print(
            f"🔍 Disease Info Node - Query: {state['query']}, Disease Name: {state.get('disease_name', 'None')}"
        )
        if state.get("disease_name"):
            query = state["disease_name"]
        else:
            query = state["query"]

        print(f"🔍 Disease Info Query: {query}")
        raw_result = self.DISEASE_INFO_AGENT.invoke(query)
        print(f"🔍 Disease Info Raw Result: {raw_result[:200]}...")  # First 200 chars
        return {
            "raw_result": raw_result,
            "query": state["query"],
            "decision": state.get("decision", ""),
            "disease_name": state.get("disease_name", ""),
        }

    def formatter_node(self, state: AgentState) -> AgentState:
        print(f"🔍 Formatter Node - All data received:")
        print(f"   Query: {state['query']}")
        print(f"   Disease Name: {state.get('disease_name', 'None')}")
        print(f"   Decision: {state.get('decision', 'None')}")
        print(f"   Raw Result Length: {len(state.get('raw_result', ''))}")

        formatted_result = self.FORMATTER_AGENT.invoke(
            query=state["query"],
            disease_name=state.get("disease_name", ""),
            raw_result=state.get("raw_result", ""),
            decision=state.get("decision", ""),
        )
        print(f"🔍 Formatter Output: {formatted_result[:200]}...")
        return {
            "formatted_result": formatted_result,
            "query": state["query"],
            "disease_name": state.get("disease_name", ""),
            "raw_result": state.get("raw_result", ""),
            "decision": state.get("decision", ""),
        }

    def main(self, q: str):
        final_state = self.app.invoke({"query": q})
        return final_state["formatted_result"]  # Return the final formatted result


if __name__ == "__main__":
    load_dotenv()
    import os

    llm_instnce = set_llm(
        os.getenv("TEST_API_KEY"), os.getenv("TEST_API_BASE"), os.getenv("TEST_MODEL")
    )
    orchestrator = AgentOrchestration(llm_instnce)

    # Test each agent separately
    def test_agents_individually():
        print("🧪 Testing Decider Agent:")
        decider_result = orchestrator.DECIDER_AGENT.invoke(
            "what should i do at my home if i got chicken pox?"
        )
        print(f"Decider: {decider_result}")

        print("\n🧪 Testing Symptom Agent:")
        symptom_result = orchestrator.SYMPTOM_AGENT.invoke(
            "what should i do at my home if i got chicken pox?"
        )
        print(f"Symptom: {symptom_result}")

        print("\n🧪 Testing Disease Info Agent:")
        disease_result = orchestrator.DISEASE_INFO_AGENT.invoke("chicken pox")
        print(f"Disease Info: {disease_result}")

        print("\n🧪 Testing Formatter Agent:")
        formatter_result = orchestrator.FORMATTER_AGENT.invoke(
            query="what should i do at my home if i got chicken pox?",
            disease_name="Chicken Pox",
            raw_result=disease_result,
            decision="disease_info",
        )
        print(f"Formatter: {formatter_result}")

    test_agents_individually()

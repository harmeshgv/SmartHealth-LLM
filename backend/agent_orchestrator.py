import logging
from typing import TypedDict, Optional, Any, List
from langgraph.graph import StateGraph, END
from PIL.Image import Image as PilImage
from langchain_core.messages import BaseMessage

from .memory import ConversationMemory
from .tool_registry import ToolRegistry
from .agents.base_agent import BaseAgent
from .agents.decider_agent import DeciderAgent
from .agents.symptom_agent import SymptomToDiseaseAgent
from .agents.disease_info_agent import DiseaseInfoAgent
from .agents.formatter_agent import FormatterAgent
from .agents.image_diagnosis_agent import ImageDiagnosisAgent
from .agents.chitchat_agent import ChitChatAgent

logger = logging.getLogger(__name__)

# This is the primary state for the main orchestrator graph
class AgentState(TypedDict):
    query: str
    image: Optional[PilImage]
    chat_history: List[BaseMessage]
    decision: str
    disease_name: Optional[str]
    raw_result: Optional[dict]
    formatted_result: str
    # Add other fields as needed for inter-agent communication

class AgentOrchestration:
    def __init__(self, llm_instance: Any, tool_registry: ToolRegistry):
        self.llm_instance = llm_instance
        self.tool_registry = tool_registry
        self.memory = ConversationMemory()
        
        # Initialize all the agents, passing their required dependencies
        self.agents: dict[str, BaseAgent] = {
            "decider": DeciderAgent(llm=llm_instance),
            "symptom_to_disease": SymptomToDiseaseAgent(llm=llm_instance, tool_registry=tool_registry),
            "disease_info": DiseaseInfoAgent(llm=llm_instance, tool_registry=tool_registry),
            "image_diagnosis": ImageDiagnosisAgent(tool_registry=tool_registry),
            "formatter": FormatterAgent(llm=llm_instance),
            "chitchat": ChitChatAgent(llm=llm_instance),
        }

        self.workflow = self._build_workflow()
        logger.info("AgentOrchestration (meta-graph) initialized with %d agents.", len(self.agents))

    def _build_workflow(self) -> StateGraph:
        """Builds the main orchestration graph."""
        workflow = StateGraph(AgentState)

        # Add nodes that correspond to invoking an agent
        workflow.add_node("decider", self.decider_node)
        workflow.add_node("symptom_to_disease", self.symptom_to_disease_node)
        workflow.add_node("disease_info", self.disease_info_node)
        workflow.add_node("image_diagnosis", self.image_diagnosis_node)
        workflow.add_node("formatter", self.formatter_node)
        workflow.add_node("chitchat", self.chitchat_node)

        workflow.set_entry_point("decider")

        # Routing logic based on the decider agent's output
        def route_decision(state: AgentState) -> str:
            decision = state.get("decision", "chitchat").lower()
            logger.debug(f"Orchestrator routing based on decision: '{decision}'")
            
            if "image_analysis" in decision and state.get("image"):
                return "image_diagnosis"
            elif "symptom" in decision:
                return "symptom_to_disease"
            elif "disease_info" in decision:
                return "disease_info"
            else: # Default to chitchat for safety
                return "chitchat"

        workflow.add_conditional_edges("decider", route_decision, {
            "image_diagnosis": "image_diagnosis",
            "symptom_to_disease": "symptom_to_disease",
            "disease_info": "disease_info",
            "chitchat": "chitchat",
        })

        # Define the flow after each agent completes its task
        workflow.add_edge("image_diagnosis", "disease_info")
        workflow.add_edge("symptom_to_disease", "disease_info")
        workflow.add_edge("disease_info", "formatter")
        workflow.add_edge("formatter", END)
        workflow.add_edge("chitchat", END) # Chitchat provides a direct final answer

        return workflow.compile()

    # --- Orchestrator Nodes ---

    async def decider_node(self, state: AgentState) -> AgentState:
        """Invokes the decider agent to determine the next step."""
        logger.debug("Orchestrator: Invoking Decider Agent.")
        
        # If an image is present, we always prioritize image analysis
        if state.get("image"):
            logger.info("Image present, forcing decision to 'image_analysis'.")
            decision = "image_analysis"
        else:
            # Otherwise, use the LLM to decide
            decider_agent = self.agents["decider"]
            decision = await decider_agent.invoke(state)

        return {**state, "decision": decision}

    async def symptom_to_disease_node(self, state: AgentState) -> AgentState:
        """Invokes the symptom-to-disease agent."""
        logger.debug("Orchestrator: Invoking SymptomToDisease Agent.")
        symptom_agent = self.agents["symptom_to_disease"]
        result = await symptom_agent.invoke(state)
        return {**state, **result}

    async def disease_info_node(self, state: AgentState) -> AgentState:
        """Invokes the disease information agent."""
        logger.debug("Orchestrator: Invoking DiseaseInfo Agent.")
        disease_info_agent = self.agents["disease_info"]
        result = await disease_info_agent.invoke(state)
        return {**state, **result}

    async def image_diagnosis_node(self, state: AgentState) -> AgentState:
        """Invokes the image diagnosis agent."""
        logger.debug("Orchestrator: Invoking ImageDiagnosis Agent.")
        image_agent = self.agents["image_diagnosis"]
        result = await image_agent.invoke(state)
        return {**state, **result}

    async def formatter_node(self, state: AgentState) -> AgentState:
        """Invokes the formatter agent to generate the final user-facing response."""
        logger.debug("Orchestrator: Invoking Formatter Agent.")
        formatter_agent = self.agents["formatter"]
        formatted_result = await formatter_agent.invoke(state)
        return {**state, "formatted_result": formatted_result}

    async def chitchat_node(self, state: AgentState) -> AgentState:
        """Invokes the chitchat agent for casual conversation."""
        logger.debug("Orchestrator: Invoking ChitChat Agent.")
        chitchat_agent = self.agents["chitchat"]
        result = await chitchat_agent.invoke(state)
        return {**state, **result}

    # --- Main Entry Point ---

    async def invoke(self, user_query: str, image: Optional[PilImage] = None) -> str:
        """Main method to invoke the agent orchestration meta-graph."""
        logger.info(f"Orchestrator invoked for query: '{user_query}'")
        self.memory.add_user_message(user_query)
        
        initial_state = {
            "query": user_query,
            "image": image,
            "chat_history": self.memory.get_history(),
            "decision": "",
            "disease_name": None,
            "raw_result": None,
            "formatted_result": "",
        }
        
        final_state = await self.workflow.ainvoke(initial_state)
        logger.info("Orchestration complete.")
        
        formatted_result = final_state.get("formatted_result", "I'm sorry, but I was unable to generate a response.")
        self.memory.add_ai_message(formatted_result)
        
        return formatted_result

    def get_memory(self) -> ConversationMemory:
        """Returns the current conversation memory."""
        return self.memory

    def set_memory(self, memory: ConversationMemory):
        """Sets the conversation memory for the instance."""
        self.memory = memory
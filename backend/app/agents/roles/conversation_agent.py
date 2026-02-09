import json
import time
import logging
from app.agents.base.base_agent import BaseAgent
from app.utils.prompt_loader import load_prompt
from typing import List, TypedDict
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    user_message: str
    llm_output: str
    run_id: str


class ConversationAgent(BaseAgent):
    agent_name = "conversation_agent"

    async def llm_node(self, state: dict):
        run_id = state["run_id"]
        logger.info("Node: llm", extra={"run_id": run_id, "agent": self.agent_name})
        # get past memory
        past = await self.recall_memory(run_id=run_id, n=5)
        logger.debug(f"Recalled memory: {past}", extra={"run_id": run_id, "agent": self.agent_name})

        # format memory into text
        past_text = "\n".join(
            [f"User: {m['user_message']}\nAssistant: {m['agent_output']}" for m in past]
        )
        logger.debug(f"Formatted past text: {past_text}", extra={"run_id": run_id, "agent": self.agent_name})

        template = load_prompt("conversation_prompt.txt")

        prompt = template.replace("{{PAST}}", past_text).replace(
            "{{INPUT}}", state["user_message"]
        )
        logger.debug(f"LLM prompt: {prompt}", extra={"run_id": run_id, "agent": self.agent_name})

        raw = await self.llm_call(prompt, run_id=run_id)
        logger.debug(f"Raw LLM response: {raw}", extra={"run_id": run_id, "agent": self.agent_name})
        state["llm_output"] = raw.strip()
        return state

    async def memory_node(self, state: dict):
        run_id = state["run_id"]
        logger.info("Node: memory", extra={"run_id": run_id, "agent": self.agent_name})
        await self.save_memory(
            user_message=state["user_message"], llm_output=state["llm_output"], run_id=run_id
        )
        return state

    def build_graph(self):
        workflow = StateGraph(ConversationState)

        async def llm_wrapper(state):
            return await self.llm_node(state)

        async def memory_wrapper(state):
            return await self.memory_node(state)

        workflow.add_node("llm", llm_wrapper)
        workflow.add_node("memory", memory_wrapper)

        workflow.set_entry_point("llm")

        workflow.add_edge("llm", "memory")
        workflow.add_edge("memory", END)

        return workflow.compile()

    async def run(self, user_message: str, run_id: str, **kwargs):
        start_time = time.time()
        logger.info("Agent run started", extra={"run_id": run_id, "agent": self.agent_name})
        
        graph = self.build_graph()

        result = await graph.ainvoke({"user_message": user_message, "llm_output": "", "run_id": run_id})
        
        end_time = time.time()
        latency = end_time - start_time
        logger.info("Agent run finished", extra={"run_id": run_id, "agent": self.agent_name, "latency": latency})

        return {"llm_output": result["llm_output"]}


# Manual debug runner
if __name__ == "__main__":
    import asyncio
    from app.core.agent_context import AgentContext

    async def test():
        context = AgentContext()
        context.debug = True

        agent = ConversationAgent(context)
        result = await agent.run("Hi who are you?", run_id="debug")
        print("\nFinal Output:", result)

    asyncio.run(test())

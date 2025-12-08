import json
from app.agents.base.base_agent import BaseAgent
from langgraph.graph import StateGraph, END
from app.utils.prompt_loader import load_prompt
from typing import List, TypedDict


class DeciderState(TypedDict):
    user_message: str
    llm_output: str
    intent: str
    agents: List[str]


class DeciderAgent(BaseAgent):
    agent_name = "decider_agent"

    async def llm_node(self, state: dict):
        template = load_prompt("decider_prompt.txt")
        prompt = template.replace("{{INPUT}}", state["user_message"])

        raw = await self.llm_call(prompt)
        state["llm_output"] = raw.strip()
        return state

    async def parse_node(self, state: dict):
        raw = state.get("llm_output", "")

        try:
            parsed = json.loads(raw)
        except:
            parsed = {"intent": "fallback", "agents": ["conversation_agent"]}

        state["intent"] = parsed.get("intent", "unknown")
        state["agents"] = parsed.get("agents", ["conversation_agent"])
        return state



    def build_graph(self):
        workflow = StateGraph(DeciderState)

        # Create async wrappers for instance methods
        async def llm_wrapper(state):
            return await self.llm_node(state)

        async def parse_wrapper(state):
            return await self.parse_node(state)



        workflow.add_node("llm", llm_wrapper)
        workflow.add_node("parse", parse_wrapper)

        workflow.set_entry_point("llm")

        workflow.add_edge("llm", "parse")
        workflow.add_edge("parse", END)

        return workflow.compile()

    async def run(self, user_message: str):
        graph = self.build_graph()

        result = await graph.ainvoke({
            "user_message": user_message,
            "llm_output": "",
            "intent": "",
            "agents": []
        })

        return {
            "intent": result["intent"],
            "agents": result["agents"]
        }


# Manual debug runner
if __name__ == "__main__":
    import asyncio
    from app.core.agent_context import AgentContext

    async def test():
        context = AgentContext()
        context.debug = True

        agent = DeciderAgent(context)
        result = await agent.run("what is malaria?")
        print("\nFinal Output:", result)

    asyncio.run(test())

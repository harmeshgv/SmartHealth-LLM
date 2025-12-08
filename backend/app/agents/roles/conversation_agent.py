import json
from app.agents.base.base_agent import BaseAgent
from app.utils.prompt_loader import load_prompt
from typing import List, TypedDict
from langgraph.graph import StateGraph, END

class ConversationState(TypedDict):
  user_message: str
  llm_output: str


class ConversationAgent(BaseAgent):
  agent_name = "conversation_agent"

  async def llm_node(self, state: dict):
      # get past memory
      past = await self.recall_memory(5)

      # format memory into text
      past_text = "\n".join(
          [f"User: {m['user_message']}\nAssistant: {m['agent_output']}"
          for m in past]
      )

      template = load_prompt("conversation_prompt.txt")

      prompt = (
          template
          .replace("{{PAST}}", past_text)
          .replace("{{INPUT}}", state["user_message"])
      )

      raw = await self.llm_call(prompt)
      state["llm_output"] = raw.strip()
      return state


  async def memory_node(self, state: dict):
    await self.save_memory(
            user_message=state["user_message"],
            llm_output=state["llm_output"]
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


  async def run(self, user_message:str):
    graph = self.build_graph()

    result = await graph.ainvoke({
      "user_message": user_message,
      "llm_output": ""
    })

    return {"llm_output": result["llm_output"]}


# Manual debug runner
if __name__ == "__main__":
    import asyncio
    from app.core.agent_context import AgentContext

    async def test():
        context = AgentContext()
        context.debug = True

        agent = ConversationAgent(context)
        result = await agent.run("Hi who are you?")
        print("\nFinal Output:", result)

    asyncio.run(test())

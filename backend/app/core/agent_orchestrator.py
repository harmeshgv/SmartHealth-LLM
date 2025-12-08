from app.agents.factories.agent_factory import AgentFactory
from app.core.agent_context import AgentContext

class AgentOrchetrator:
  def __init__(self, context: AgentContext):
    self.context = context
    self.factory = AgentFactory(context)

  async def run(self, user_message: str):
      decider = self.factory.create("decider_agent")
      plan = await decider.run(user_message)

      state = {"user_message": user_message}

      # Run planned agents
      for agent_name in plan["agents"]:
          agent = self.factory.create(agent_name)
          result = await agent.run(**state)
          state.update(result)

      # If it's a medical workflow → run final reasoning
      if plan.get("type") == "medical":
          reasoner = self.factory.create("reasoning_agent")
          final = await reasoner.run(**state)
          return {"final_output": final["final_output"]}

      # If it's just conversation → return that output directly
      return {"final_output": state.get("llm_output")}





# Manual debug runner
if __name__ == "__main__":
    import asyncio
    from app.core.agent_context import AgentContext

    async def test():
        context = AgentContext(

        )

        orch = AgentOrchetrator(context)
        result = await orch.run("I have fever and vomiting")
        print("\n[FINAL OUTPUT]:")
        print(result)

    asyncio.run(test())

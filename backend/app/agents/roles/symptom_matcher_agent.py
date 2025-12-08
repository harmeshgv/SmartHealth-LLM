import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from app.agents.base.base_agent import BaseAgent
from app.tools.medical.biomedical_ner_tool import BiomedicalNERTool
from app.tools.medical.symptom_matcher_tool import SymptomDiseaseMatcherTool
from app.utils.prompt_loader import load_prompt

class SymptomMatcherState(TypedDict):
  user_message:str
  ner_entities: Dict[str, Any]
  disease_matched: str
  llm_output: str
  symptoms: List[str]

class SymptomMatcherAgent(BaseAgent):
  agent_name ="symptom_matcher_agent"

  def __init__(self, context):
    super().__init__(context)

    self.ner_tool = BiomedicalNERTool()
    self.symp_match_tool = SymptomDiseaseMatcherTool()

  async def ner_node(self, state: SymptomMatcherState):
    entities = await self.ner_tool.run(state["user_message"])
    state["ner_entities"] = entities
    return state

  async def llm_node(self, state: SymptomMatcherState):
    template = load_prompt("symptom_matcher_prompt.txt")
    prompt = (
            template
            .replace("{{INPUT}}", state["user_message"])
            .replace("{{NER_CONTEXT}}", json.dumps(state["ner_entities"]))
        )

    raw = await self.llm_call(prompt)
    state["llm_output"] = raw.strip()
    return state

  async def parse_node(self, state: SymptomMatcherState):
    raw = state.get("llm_output", "")

    try:
      parsed = json.loads(raw)
    except:
      parsed = {"symptoms" : ["None"]}

    state["symptoms"] = parsed.get("symptoms", ["None"])
    return state


  async def matcher_node(self, state: SymptomMatcherState):
      matches = await self.symp_match_tool.run(state["symptoms"])

      disease_name = None

      # Case 1: tool returns dict with a list of matches
      if isinstance(matches, dict) and "matched_diseases" in matches:
          ranked = matches["matched_diseases"]
          if isinstance(ranked, list) and len(ranked) > 0:
              disease_name = ranked[0].get("disease")  # TOP MATCH

      # Case 2: tool already returns a string
      elif isinstance(matches, str):
          disease_name = matches

      # Safety fallback
      if not disease_name:
          disease_name = "unknown"

      state["disease_matched"] = disease_name
      return state


  async def memory_node(self, state: dict):
    await self.save_memory(
            user_message=state["user_message"],
            llm_output=state["llm_output"]
    )
    return state


  def build_graph(self):
    workflow = StateGraph(SymptomMatcherState)

    async def ner_wrapper(state):
      return await self.ner_node(state)

    async def llm_wrapper(state):
      return await self.llm_node(state)

    async def parse_wrapper(state):
      return await self.parse_node(state)

    async def matcher_wrapper(state):
      return await self.matcher_node(state)

    async def memory_wrapper(state):
      return await self.memory_node(state)

    workflow.add_node("ner", ner_wrapper)
    workflow.add_node("llm", llm_wrapper)
    workflow.add_node("parse", parse_wrapper)
    workflow.add_node("matcher", matcher_wrapper)
    workflow.add_node("memory", memory_wrapper)

    workflow.set_entry_point("ner")

    workflow.add_edge("ner", "llm")
    workflow.add_edge("llm", "parse")
    workflow.add_edge("parse", "matcher")
    workflow.add_edge("matcher", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()

  async def run(self, user_message:str):
    graph = self.build_graph()

    result = await graph.ainvoke({
      "user_message": user_message,
      "ner_entities": {},
      "disease_matched" : "",
      "llm_output" : "",
      "symptoms": []
    })

    return {
    "symptoms": result["symptoms"],
    "disease_matched": result["disease_matched"]
}


# Manual debug runner
if __name__ == "__main__":
    import asyncio
    from app.core.agent_context import AgentContext

    async def test():
        context = AgentContext()
        context.debug = True

        agent = SymptomMatcherAgent(context)
        result = await agent.run("shock, nausea, joint pain, organ failure, headache, nausea\nvomiting\npain, restlessness, severe stomach pain, fatigue, bleeding from nose, rapid breathing, stomach pain, dengue hemorrhagic fever, death, blood in urine, rash, difficult breathing, vomiting, vomiting\nbleeding, bleeding, bleeding under the skin, dengue infection, pain behind the eyes, bleeding from gums, muscle pain, irritability, dengue, fever, dengue shock syndrome, blood in stools, vomit, swollen glands, bone pain, blood in vomit, dengue fever, bruising'}}}")
        print("\nFinal Output:", result)

    asyncio.run(test())

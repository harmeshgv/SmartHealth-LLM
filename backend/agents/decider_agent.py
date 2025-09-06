import os
import sys
import json
from gravixlayer import GravixLayer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.utils.llm import LLM
from backend.agents.symptom_agent import DiseaseMatcher
from backend.agents.disease_info_gent import DISEASEINFOAGENT


class DECIDERAGENT:
    def __init__(self, model="meta-llama/llama-3.1-8b-instruct"):
        self.llm = LLM()
        self.diseaseinfoagent = DISEASEINFOAGENT()
        self.symtomtodiseaseagent = DiseaseMatcher()
        self.tools = [
            {
                "name": "symptom_to_disease",
                "description": "Takes a list of symptoms and returns the most likely disease.",
                "parameters": {
                    "symptoms": "list of symptoms (strings)"
                }
            },
            {
                "name": "disease_info",
                "description": "Takes a disease name and returns information about it.",
                "parameters": {
                    "disease_name": "string"
                }
            }
        ]

        self.tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tools]
        )

    def build_prompt(self, query):
        return f"""
        You are a decision-making agent.
        Tools available:
        {self.tool_descriptions}

        User query: "{query}"

        Decide the BEST tool to use and extract the parameters.

        Return STRICTLY in this JSON format (no extra text!):
        {{
            "agent": "tool_name",
            "parameters": {{
                "param1": "value",
                "param2": ["value1", "value2"]
            }}
        }}

        Example 1:
        User query: "I have fever and cough"
        {{
            "agent": "symptom_to_disease",
            "parameters": {{
                "symptoms": ["fever", "cough"]
            }}
        }}

        Example 2:
        User query: "Tell me about diabetes"
        {{
            "agent": "disease_info",
            "parameters": {{
                "disease_name": "diabetes"
            }}
        }}
        """
    
    def build_final_prompt(self, info, query):
        return f"""
        You are medical info to human type langage convertor.
        So with this info you have {info} answer the following query {query}"""

    def main(self, query):
        prompt = self.build_prompt(query)
        decision = self.llm.talk(prompt)
        decision = json.loads(decision)
        if decision['agent'] == "disease_info":
            info = self.diseaseinfoagent.match(decision['parameters']['disease_name'])
        else:
            info = self.symtomtodiseaseagent.match(decision['parameters']['symptoms'])
        

        final = self.llm.talk(self.build_final_prompt(info, query))
        print(final)
        return final




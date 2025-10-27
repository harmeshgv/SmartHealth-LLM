# backend/agents/symptom_to_disease_agent.py

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional, List

from tools.disease_matcher_tool import match_disease_tool
from tools.biomedical_ner_tool import extract_data


# ---------------- Schema ----------------
class SymptomState(TypedDict):
    input: str
    symptoms: Optional[list]
    chat_history: list
    user_query: str
    entity_context: list
    matched_diseases: Optional[list]
    final_disease: Optional[str]
    response: Optional[str]


class SymptomToDiseaseAgent:
    def __init__(self, llm, name: str = "Symptom to Disease Agent"):
        self.name = name
        self.llm = llm
        self.graph = self._build_graph()

    # ---------------- Prompts ----------------
    def _build_symptom_extraction_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical symptom extractor. Extract ALL symptoms mentioned by the user and return them as a comma-separated list.

    CRITICAL: You MUST return symptoms even if they are described in everyday language.

    Examples:
    - "I have fever and headache" → "fever, headache"
    - "My throat hurts and I'm coughing" → "sore throat, cough"
    - "Feeling nauseous with body pain" → "nausea, body pain"
    - "Head is pounding and stomach feels upset" → "headache, nausea"
    - "Can't stop sneezing and nose is running" → "sneezing, runny nose"
    - "Feeling hot and shivery" → "fever, chills"

    Rules:
    1. Extract ALL symptoms mentioned
    2. Convert everyday language to standard symptom names
    3. Return ONLY comma-separated symptoms, no other text
    4. If no symptoms, return "none"

    User Input: {user_input}

    Symptoms:""",
                )
            ]
        )

    def _build_disease_decision_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical assistant. Your ONLY task is to identify the SINGLE most likely disease based on the given symptoms and matches.

            Available Disease Matches: {matched_diseases}
            User Symptoms: {symptoms}

            CRITICAL RULES:
            - Return ONLY the disease name as plain text
            - NO explanations, NO probabilities, NO multiple diseases
            - JUST the disease name, nothing else
            - If no good match, return "unknown"

            Disease:""",
                )
            ]
        )

    def _build_final_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful medical assistant. Provide a clear, reassuring response about the possible condition.

            User Query: {user_query}
            Identified Disease: {final_disease}
            Symptoms: {symptoms}

            Keep it brief, empathetic, and suggest seeing a doctor.""",
                ),
                ("human", "Please provide a helpful response:"),
            ]
        )

    # ---------------- Graph Nodes ----------------
    def _extract_entities(self, state):
        """Extract entity context from input"""
        try:
            ner_result = extract_data({"input": state["input"]})
            return {
                **state,
                "entity_context": ner_result.get("entity_context", []),
                "has_medical_context": ner_result.get("has_medical_context", False),
            }
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return {**state, "entity_context": [], "has_medical_context": False}

    def _extract_symptoms(self, state):
        """Extract symptoms using LLM"""
        try:
            entity_context = state.get("entity_context", [])
            user_query = state["user_query"]

            print(f"DEBUG - User query: {user_query}")
            print(f"DEBUG - Entity context: {entity_context}")

            # First try to extract from entity context
            symptoms_from_context = []
            for entity in entity_context:
                if entity.startswith("symptom:"):
                    symptom = entity.replace("symptom:", "").strip()
                    symptoms_from_context.append(symptom)

            if symptoms_from_context:
                symptoms_str = ", ".join(symptoms_from_context)
                print(f"DEBUG - Extracted symptoms from context: {symptoms_str}")
                return {**state, "symptoms": symptoms_from_context}

            # If no symptoms in context, use LLM with the actual user query
            prompt = self._build_symptom_extraction_prompt()
            messages = prompt.invoke(
                {"user_input": user_query}
            )  # Changed from "symptoms" to "user_input"
            response = self.llm.invoke(messages)

            if hasattr(response, "content"):
                symptoms_str = response.content.strip()
            else:
                symptoms_str = str(response).strip()

            print(f"DEBUG - LLM raw response: '{symptoms_str}'")

            # Handle "none" response
            if symptoms_str.lower() == "none":
                print("DEBUG - LLM returned 'none' for symptoms")
                return {**state, "symptoms": []}

            # Parse comma-separated symptoms
            symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
            print(f"DEBUG - LLM extracted symptoms: {symptoms}")

            return {**state, "symptoms": symptoms}

        except Exception as e:
            print(f"Error extracting symptoms: {e}")
            return {**state, "symptoms": []}

    def _match_symptom_to_disease(self, state):
        """Match symptoms to diseases using the tool"""
        try:
            symptoms = state.get("symptoms", [])
            if not symptoms:
                return {**state, "matched_diseases": [], "final_disease": "unknown"}

            # Use the correct function call
            matches_result = match_disease_tool({"symptoms": symptoms})
            print(f"DEBUG - Disease matches result: {matches_result}")

            # Extract the matched diseases from the result
            matched_diseases = matches_result.get("matched_diseases", [])
            return {**state, "matched_diseases": matched_diseases}

        except Exception as e:
            print(f"Error matching symptoms to disease: {e}")
            return {**state, "matched_diseases": [], "final_disease": "unknown"}

    # In your SymptomToDiseaseAgent, update the _decide_final_disease method:

    def _decide_final_disease(self, state):
        """Decide on the final disease using LLM - return ONLY disease name"""
        try:
            matched_diseases = state.get("matched_diseases", [])
            symptoms = state.get("symptoms", [])

            if not matched_diseases or not symptoms:
                return {**state, "final_disease": "unknown"}

            # Create a simpler, more direct prompt
            disease_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """EXTRACT ONLY THE DISEASE NAME from the matched diseases.

    Matched Diseases: {matched_diseases}
    User Symptoms: {symptoms}

    Return ONLY the disease name, nothing else. Examples:
    - "Common cold"
    - "Influenza"
    - "Migraine"
    - If unsure, return "unknown"

    Disease name:""",
                    )
                ]
            )

            messages = disease_prompt.invoke(
                {
                    "matched_diseases": str(matched_diseases),
                    "symptoms": ", ".join(symptoms),
                }
            )
            response = self.llm.invoke(messages)

            if hasattr(response, "content"):
                disease = response.content.strip()
            else:
                disease = str(response).strip()

            # Clean up the response - extract only the disease name
            disease = disease.split(".")[0]  # Take only first sentence
            disease = disease.replace('"', "").replace("'", "").strip()

            # If it's still a long response, try to extract just the disease name
            if len(disease.split()) > 3:
                # Look for common disease patterns
                for match in matched_diseases:
                    db_disease = match.get("disease", "")
                    if db_disease.lower() in disease.lower():
                        disease = db_disease
                        break

            print(f"DEBUG - Cleaned disease name: {disease}")
            return {**state, "final_disease": disease}

        except Exception as e:
            print(f"Error deciding final disease: {e}")
            return {**state, "final_disease": "unknown"}

    def _generate_final_response(self, state):
        """Generate final response to user - return ONLY disease name"""
        try:
            final_disease = state.get("final_disease", "unknown")

            # For the orchestration, we ONLY want the disease name
            # The formatting will be done by the formatter agent
            if final_disease == "unknown":
                return {**state, "response": "unknown"}
            else:
                # Return just the disease name, not a full response
                return {**state, "response": final_disease}

        except Exception as e:
            print(f"Error generating final response: {e}")
            return {**state, "response": "unknown"}

    # ---------------- Graph ----------------
    def _build_graph(self):
        graph = StateGraph(SymptomState)

        # Add nodes
        graph.add_node("ner", RunnableLambda(self._extract_entities))
        graph.add_node("extract_symptoms", RunnableLambda(self._extract_symptoms))
        graph.add_node("map_disease", RunnableLambda(self._match_symptom_to_disease))
        graph.add_node("decide_disease", RunnableLambda(self._decide_final_disease))
        graph.add_node("final", RunnableLambda(self._generate_final_response))

        # Set entry point
        graph.set_entry_point("ner")

        # Define edges
        graph.add_edge("ner", "extract_symptoms")
        graph.add_edge("extract_symptoms", "map_disease")
        graph.add_edge("map_disease", "decide_disease")
        graph.add_edge("decide_disease", "final")
        graph.add_edge("final", END)

        return graph.compile()

    def invoke(self, user_input: str) -> str:
        """Invoke the symptom to disease agent"""
        try:
            result = self.graph.invoke(
                {
                    "input": user_input,
                    "user_query": user_input,
                    "chat_history": [],
                    "symptoms": None,
                    "entity_context": [],
                    "matched_diseases": None,
                    "final_disease": None,
                    "response": None,
                }
            )

            # Return the final response
            if hasattr(result.get("response"), "content"):
                return result["response"].content
            elif isinstance(result.get("response"), str):
                return result["response"]
            else:
                return str(result.get("response", "No response generated"))

        except Exception as e:
            return f"Error: {str(e)}"


# ---------------- Test ----------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from backend.utils.llm import set_llm

    load_dotenv()
    llm_instance = set_llm(
        os.getenv("TEST_API_KEY"),
        os.getenv("TEST_API_BASE"),
        os.getenv("TEST_MODEL"),
    )

    agent = SymptomToDiseaseAgent(llm_instance)

    # Test cases
    test_queries = [
        "I have fever and cough",
        "My head hurts and I feel nauseous",
        "Sore throat and runny nose",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        result = agent.invoke(query)
        print(f"Response: {result}")
        print(f"{'='*50}")

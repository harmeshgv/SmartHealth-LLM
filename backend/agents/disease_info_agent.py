from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Optional
import json

from tools.biomedical_ner_tool import extract_data
from tools.disease_info_retriever_tool import retrieve_disease_info
from tools.google_search import google_search


# ---------------- Schema ----------------
class DiseaseState(TypedDict):
    input: str
    disease: Optional[str]
    info: Optional[str]
    decision: Optional[str]
    chat_history: list
    user_query: str
    entity_context: list
    extracted_disease: Optional[str]


# ---------------- Agent ----------------
class DiseaseInfoAgent:
    def __init__(self, llm, name: str = "Disease Info Agent"):
        self.name = name
        self.llm = llm
        self.graph = self._build_graph()

    # ---------------- Prompts ----------------
    def _build_disease_extraction_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a medical entity resolver. Extract the specific disease/condition name from the user query and NER context.

                NER Context: {entity_context}
                User Query: {user_query}

                Return ONLY the disease name as a simple string. If no clear disease is mentioned, return "unknown".

                IMPORTANT: When you see disease/condition entities in the NER context, use those as the primary source.

                Examples:
                - Context: ["disease/condition: malaria"], Query: "I want to know about malaria" → "malaria"
                - Context: ["symptom: headache", "symptom: fever"], Query: "I have flu symptoms" → "influenza"
                - Context: ["disease/condition: diabetes"], Query: "tell me about diabetes" → "diabetes"
                - Context: ["symptom: cough", "symptom: sore throat"], Query: "common cold info" → "common cold"
                - Context: [], Query: "I don't feel well" → "unknown"

                Disease name:""",
                )
            ]
        )

    def _build_synth_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a tool decider. Analyze if the database information is sufficient to answer the user's query.

                Database Information: {info}
                User Query: {user_query}

                Return ONLY one word: either 'search' or 'ok'.

                Return 'ok' if:
                - The database provides comprehensive information about the disease
                - The information covers symptoms, causes, treatment, and prevention
                - The response adequately answers the user's query

                Return 'search' only if:
                - The database returned an error or no information
                - The information is extremely brief (less than 2-3 sentences)
                - Key aspects like symptoms or treatment are completely missing

                Your response must be exactly one word: 'search' or 'ok'.""",
                ),
                ("human", "User query: {user_query}\n\nDatabase information: {info}"),
            ]
        )

    def _build_final_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise medical assistant. Provide clear, accurate information about the disease based on the available information. Be helpful but concise.",
                ),
                (
                    "human",
                    "User question: {user_query}\n\nAvailable information: {info}",
                ),
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

    def _extract_disease_name(self, state):
        """Use LLM to extract specific disease name from context"""
        try:
            entity_context = state.get("entity_context", [])
            user_query = state["user_query"]

            print(f"DEBUG - Entity Context: {entity_context}")
            print(f"DEBUG - User Query: {user_query}")

            # First, try to extract disease directly from entity context
            for entity in entity_context:
                if entity.startswith("disease/condition:"):
                    disease_name = entity.replace("disease/condition:", "").strip()
                    print(f"DEBUG - Found disease in context: {disease_name}")
                    return {**state, "extracted_disease": disease_name}

            # If no direct disease entity found, use LLM
            if not state.get("has_medical_context"):
                return {**state, "extracted_disease": "unknown"}

            prompt = self._build_disease_extraction_prompt()
            messages = prompt.invoke(
                {"entity_context": entity_context, "user_query": user_query}
            )

            response = self.llm.invoke(messages)

            # Extract disease name from response
            if hasattr(response, "content"):
                disease_name = response.content.strip().lower()
            else:
                disease_name = str(response).strip().lower()

            print(f"DEBUG - LLM raw response: '{disease_name}'")

            # Clean up the response - more aggressive cleaning
            disease_name = disease_name.replace('"', "").replace("'", "").strip()

            # Remove common prefixes
            prefixes = [
                "disease name:",
                "disease:",
                "condition:",
                "the disease is",
                "it is",
            ]
            for prefix in prefixes:
                if disease_name.startswith(prefix):
                    disease_name = disease_name.replace(prefix, "").strip()

            # If still "unknown" but we have disease entities, use the first one
            if disease_name == "unknown" and entity_context:
                for entity in entity_context:
                    if "disease/condition" in entity:
                        disease_name = (
                            entity.split(":")[1].strip() if ":" in entity else entity
                        )
                        print(f"DEBUG - Overriding 'unknown' with: {disease_name}")
                        break

            print(f"DEBUG - Final disease name: {disease_name}")
            return {**state, "extracted_disease": disease_name}

        except Exception as e:
            print(f"Error extracting disease name: {e}")
            # Fallback: try to extract from entity context
            entity_context = state.get("entity_context", [])
            for entity in entity_context:
                if "disease/condition" in entity:
                    disease_name = (
                        entity.split(":")[1].strip() if ":" in entity else entity
                    )
                    return {**state, "extracted_disease": disease_name}
            return {**state, "extracted_disease": "unknown"}

    def _retrieve_disease_info(self, state):
        """Retrieve disease information using extracted disease name"""
        try:
            disease_name = state.get("extracted_disease")

            if disease_name and disease_name != "unknown":
                print(f"Looking up disease: '{disease_name}'")

                info = retrieve_disease_info({"disease": disease_name})
                print(f"Database returned valid info: {info is not None}")

                return {**state, "info": info, "disease": disease_name}
            else:
                return {
                    **state,
                    "info": "No specific disease identified.",
                    "disease": "unknown",
                }

        except Exception as e:
            print(f"Error retrieving disease info: {e}")
            return {**state, "info": None}

    def _make_decision(self, state):
        """Make decision based on available information"""
        try:
            info = state.get("info")

            print(f"=== DECISION MAKING ===")
            print(f"Info type: {type(info)}")

            # If we have no info at all, search
            if not info:
                print("Decision: No info available - searching")
                return {**state, "decision": "search"}

            # If info is a string error message, search
            if isinstance(info, str) and "no specific disease" in info.lower():
                print("Decision: No disease identified - searching")
                return {**state, "decision": "search"}

            # If we have database info with error, search
            if isinstance(info, dict):
                db_info = info.get("info", {})
                if isinstance(db_info, dict) and "error" in db_info:
                    print("Decision: Database error - searching")
                    return {**state, "decision": "search"}

                # Check if we have substantial content
                if isinstance(db_info, dict) and db_info.get("Overview"):
                    overview = db_info["Overview"]
                    word_count = len(overview.split())
                    print(f"Overview word count: {word_count}")
                    if word_count > 50:
                        print("Decision: Good database info - using it (OK)")
                        return {**state, "decision": "ok"}

            # Default to search if we're unsure
            print("Decision: Defaulting to search")
            return {**state, "decision": "search"}

        except Exception as e:
            print(f"Error in decision making: {e}")
            return {**state, "decision": "search"}

    def _perform_search(self, state):
        """Perform web search if needed"""
        try:
            if state.get("decision") == "search":
                search_query = state.get("disease", state["input"])
                search_results = google_search(search_query)

                current_info = state.get("info", "")
                combined_info = (
                    f"{current_info}\n\nAdditional Information:\n{search_results}"
                    if current_info
                    else search_results
                )
                return {**state, "info": combined_info}
            return state
        except Exception as e:
            print(f"Error in search: {e}")
            return state

    def _generate_final_response(self, state):
        """Generate final response to user"""
        try:
            prompt = self._build_final_prompt()
            messages = prompt.invoke(
                {
                    "user_query": state["user_query"],
                    "info": state.get("info", "No information available"),
                }
            )
            response = self.llm.invoke(messages)
            return {**state, "response": response}
        except Exception as e:
            print(f"Error generating final response: {e}")
            return {
                **state,
                "response": "Sorry, I encountered an error processing your request.",
            }

    # ---------------- Graph ----------------
    def _build_graph(self):
        graph = StateGraph(DiseaseState)

        # Add nodes - REMOVE synth_decision
        graph.add_node("ner", RunnableLambda(self._extract_entities))
        graph.add_node("extract_disease", RunnableLambda(self._extract_disease_name))
        graph.add_node("db_lookup", RunnableLambda(self._retrieve_disease_info))
        graph.add_node(
            "decision", RunnableLambda(self._make_decision)
        )  # Simple decision
        graph.add_node("search", RunnableLambda(self._perform_search))
        graph.add_node("final", RunnableLambda(self._generate_final_response))

        # Set entry point
        graph.set_entry_point("ner")

        # Define edges - go directly to decision after db_lookup
        graph.add_edge("ner", "extract_disease")
        graph.add_edge("extract_disease", "db_lookup")
        graph.add_edge("db_lookup", "decision")  # Direct to simple decision

        # Conditional routing
        graph.add_conditional_edges(
            "decision",
            lambda state: state.get("decision", "search"),
            {
                "ok": "final",
                "search": "search",
            },
        )

        graph.add_edge("search", "final")
        graph.add_edge("final", END)

        return graph.compile()

    # ---------------- Invoke ----------------
    def invoke(self, user_input: str):
        try:
            result = self.graph.invoke(
                {
                    "input": user_input,
                    "chat_history": [],
                    "user_query": user_input,
                    "disease": None,
                    "info": None,
                    "decision": None,
                    "entity_context": [],
                    "extracted_disease": None,
                }
            )
            # Return the final response
            if hasattr(result.get("response"), "content"):
                return result["response"].content
            elif isinstance(result.get("response"), str):
                return result["response"]
            else:
                return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


# ---------------- Run Test ----------------
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
    DI = DiseaseInfoAgent(llm_instance)
    class_names = [
        "cellulitis",
        "BA-impetigo",
        "FU-athlete-foot",
        "FU-nail-fungus",
        "FU-ringworm",
        "PA-cutaneous-larva-migrans",
        "VI-chickenpox",
        "VI-shingles",
    ]
    print(DI.invoke(class_names[0]))

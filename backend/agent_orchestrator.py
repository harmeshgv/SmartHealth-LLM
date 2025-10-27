from langgraph.graph import StateGraph, END
from agents.decider_agent import DeciderAgent
from agents.disease_info_agent import DiseaseInfoAgent
from agents.symptom_agent import SymptomToDiseaseAgent
from agents.formatter_agent import FormatterAgent
from agents.image_diagnosis_agent import SkinDiseasePrediction
from typing import TypedDict, Optional
from PIL.Image import Image as PilImage


class AgentState(TypedDict):
    query: str
    decision: str  # "disease_info", "symptom_to_disease", or "image_analysis"
    disease_name: str
    raw_result: str  # Raw result from disease info agent
    formatted_result: str  # Final formatted result
    symptoms: Optional[list]  # For symptom agent
    matched_diseases: Optional[list]  # For symptom agent
    image: Optional[PilImage]  # Image for analysis


class AgentOrchestration:
    def __init__(self, llm_instance):
        self.llm_instance = llm_instance

        # Instantiate all agents including formatter
        self.DECIDER_AGENT = DeciderAgent(self.llm_instance)
        self.SYMPTOM_AGENT = SymptomToDiseaseAgent(self.llm_instance)
        self.DISEASE_INFO_AGENT = DiseaseInfoAgent(self.llm_instance)
        self.FORMATTER_AGENT = FormatterAgent(self.llm_instance)
        self.SKIN_DISEASE_PREDICTION_AGENT = SkinDiseasePrediction()

        # Initialize workflow
        self.workflow = StateGraph(AgentState)

        # Add all nodes including formatter
        self.workflow.add_node("decider", self.decider_node)
        self.workflow.add_node("symptom_to_disease", self.symptom_node)
        self.workflow.add_node("disease_info", self.disease_info_node)
        self.workflow.add_node("formatter", self.formatter_node)
        self.workflow.add_node("skin_disease_prediction", self.skin_disease_node)

        # Set entrypoint
        self.workflow.set_entry_point("decider")

        # Define conditional routing
        def route_decision(state: AgentState) -> str:
            decision = state.get("decision", "").lower()
            has_image = state.get("image") is not None
            print(f"🔄 Routing decision: {decision}, Has image: {has_image}")

            # If there's an image, prioritize image analysis
            if has_image:
                return "skin_disease_prediction"
            elif "symptom_to_disease" in decision:
                return "symptom_to_disease"
            else:  # Default to disease_info
                return "disease_info"

        # Add conditional edges
        self.workflow.add_conditional_edges(
            "decider",
            route_decision,
            {
                "disease_info": "disease_info",
                "symptom_to_disease": "symptom_to_disease",
                "skin_disease_prediction": "skin_disease_prediction",
            },
        )

        # From skin disease prediction, go to disease info to get details
        self.workflow.add_edge("skin_disease_prediction", "disease_info")

        # From symptom agent, go to disease info to get details
        self.workflow.add_edge("symptom_to_disease", "disease_info")

        # From disease info, always go to formatter
        self.workflow.add_edge("disease_info", "formatter")

        # Formatter ends the workflow
        self.workflow.add_edge("formatter", END)

        # Compile workflow
        self.app = self.workflow.compile()
        print("AgentOrchestration initialized")

    def decider_node(self, state: AgentState) -> AgentState:
        """Decide which agent should handle the query"""
        print(f"🔍 Decider Node - Query: {state['query']}")

        # If there's an image, we'll handle it in the routing
        if state.get("image"):
            print("🖼️ Image detected in decider node")
            return {
                **state,
                "decision": "image_analysis",
                "disease_name": "",
                "raw_result": "",
            }

        decision = self.DECIDER_AGENT.invoke(state["query"])
        print(f"✅ Decider Decision: {decision}")

        return {
            **state,
            "decision": decision,
            "disease_name": "",  # Reset for new flow
            "raw_result": "",  # Reset for new flow
        }

    def _extract_disease_name_from_symptoms(self, query: str) -> str:
        """Fallback method to extract disease name directly"""
        try:
            # Simple direct extraction for common symptoms
            symptom_mapping = {
                "fever and cough": "Common cold",
                "headache and nausea": "Migraine",
                "sore throat and runny nose": "Common cold",
                "fever": "Influenza",
                "cough": "Bronchitis",
            }

            query_lower = query.lower()
            for symptoms, disease in symptom_mapping.items():
                if symptoms in query_lower:
                    return disease

            return "unknown"
        except:
            return "unknown"

    def symptom_node(self, state: AgentState) -> AgentState:
        """Process symptom queries and extract disease name"""
        print(f"🤒 Symptom Node - Query: {state['query']}")

        # Use the symptom agent to get disease name
        disease_name = self.SYMPTOM_AGENT.invoke(state["query"])
        print(f"✅ Symptom Agent Output Disease: {disease_name}")

        # Clean the disease name - ensure it's just a name, not a full response
        if (
            "sorry" in disease_name.lower()
            or "consult" in disease_name.lower()
            or len(disease_name) > 50
        ):
            print("⚠️  Cleaning disease name - too conversational")
            # Extract just the disease name from the matched diseases
            disease_name = self._extract_disease_name_from_symptoms(state["query"])

        return {
            **state,
            "disease_name": disease_name,
        }

    def disease_info_node(self, state: AgentState) -> AgentState:
        """Get disease information using either direct query or extracted disease name"""
        print(f"📚 Disease Info Node - Query: {state['query']}")
        print(
            f"📚 Disease Name from previous step: {state.get('disease_name', 'None')}"
        )

        # Determine what to search for
        search_query = ""
        if state.get("disease_name") and state["disease_name"] != "unknown":
            # Use the disease name extracted by symptom agent or image prediction
            search_query = state["disease_name"]
            print(f"🔍 Searching with disease name: {search_query}")
        else:
            # Use the original query
            search_query = state["query"]
            print(f"🔍 Searching with original query: {search_query}")

        # Get disease information
        raw_result = self.DISEASE_INFO_AGENT.invoke(search_query)
        print(f"✅ Disease Info Raw Result Length: {len(str(raw_result))} chars")

        return {
            **state,
            "raw_result": raw_result,
        }

    def formatter_node(self, state: AgentState) -> AgentState:
        """Format the final response"""
        formatted_result = self.FORMATTER_AGENT.invoke(
            query=state["query"],
            disease_name=state.get("disease_name", ""),
            raw_result=state.get("raw_result", ""),
            decision=state.get("decision", ""),
        )
        print(f"✅ Formatter Output Ready ({len(formatted_result)} chars)")

        return {
            **state,
            "formatted_result": formatted_result,
        }

    def skin_disease_node(self, state: AgentState) -> AgentState:
        """Predict disease from image if provided in the state"""
        image = state.get("image")
        if image:
            print("🖼️ Processing image in skin disease node...")
            try:
                # Define class names for the model
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

                # Save image temporarily and predict
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_file:
                    image.save(temp_file.name, format="JPEG")
                    predicted_class = self.SKIN_DISEASE_PREDICTION_AGENT.predict_image(
                        temp_file.name, class_names
                    )
                    os.unlink(temp_file.name)  # Clean up temp file

                print(f"✅ Predicted disease from image: {predicted_class}")

                state["disease_name"] = predicted_class
                state["decision"] = "image_analysis"

            except Exception as e:
                print(f"❌ Error in image prediction: {e}")
                state["disease_name"] = "unknown skin condition"
                state["raw_result"] = f"Unable to analyze the image: {str(e)}"
        else:
            print("❌ No image found in skin disease node")
            state["disease_name"] = "unknown"

        return state

    def invoke(self, user_query: str, image: Optional[PilImage] = None) -> str:
        """Main method to invoke the orchestration"""
        print(f"\n🚀 Starting Agent Orchestration for: '{user_query}'")
        print(f"🖼️ Image provided: {image is not None}")
        print("=" * 60)

        try:
            # Initialize state
            initial_state = {
                "query": user_query,
                "decision": "",
                "disease_name": "",
                "raw_result": "",
                "formatted_result": "",
                "symptoms": None,
                "matched_diseases": None,
                "image": image,  # Pass image here
            }

            # Execute the workflow
            final_state = self.app.invoke(initial_state)
            print("=" * 60)
            print("🎉 Agent Orchestration Completed Successfully!")

            return final_state["formatted_result"]

        except Exception as e:
            error_msg = f"❌ Error in agent orchestration: {str(e)}"
            print(error_msg)
            return error_msg

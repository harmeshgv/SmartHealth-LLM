from tools.skin_disease_prediction_tool import SkinDiseasePrediction
from utils.utils import read_json

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Optional
from PIL.Image import Image as PilImage


class DiseaseState(TypedDict):
    input: str
    image_path: Optional[PilImage]
    disease: Optional[str]
    info: Optional[str]
    decision: Optional[str]
    chat_history: list
    user_query: str
    entity_context: list
    extracted_disease: Optional[str]


class SkinDiseasePredictionAgent:
    def __init__(self, class_label_path: str = "backend/data/labels.json"):
        self.SDP = SkinDiseasePrediction()
        self.classes = read_json(class_label_path)
        self.graph = self._build_graph()

    def _predict_disease(self, state: DiseaseState):
        if state["image"]:
            predicted_class = self.SDP.predict_image(
                image_path=state["image"], class_names=self.classes
            )
            state["disease"] = predicted_class  # assign directly
        return state

    def _build_graph(self):
        graph = StateGraph(DiseaseState)
        graph.add_node("DenseNet", RunnableLambda(self._predict_disease))

        graph.set_entry_point("DenseNet")

        graph.add_edge("DenseNet", END)

        return graph.compile()

    def invoke(self, user_input: str, image: PilImage):
        try:
            result = self.graph.invoke(
                {
                    "input": user_input,
                    "image_path": image,
                    "chat_history": [],
                    "user_query": user_input,
                    "disease": None,
                    "info": None,
                    "decision": None,
                    "entity_context": [],
                    "extracted_disease": None,
                }
            )

            if hasattr(result.get("response"), "content"):
                return result["response"].content
            elif isinstance(result.get("response"), str):
                return result["response"]
            else:
                return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    SDPA = SkinDiseasePredictionAgent()
    print(SDPA.invoke("itching", "backend/agents/download.jpeg"))

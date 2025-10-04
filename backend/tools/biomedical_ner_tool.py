# backend/tools/biomedical_ner_tool.py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain.tools import tool
from typing import List


class BiomedicalNER:
    def __init__(self, model_path: str = "SmartHealth/biomedicalnerall"):
        self.tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
        self.model = AutoModelForTokenClassification.from_pretrained(
            "d4data/biomedical-ner-all"
        )
        self.pipe = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="average",
        )

    def extract_entities(self, text: str) -> List[dict]:
        entities = self.pipe(text)
        return entities


ner = BiomedicalNER()


@tool
def extract_data(text: str) -> str:
    """
    Extracts biomedical entities from the input text using a pre-trained NER model. This helps to identify
    relevant biomedical terms such as diseases, medications, symptoms, and procedures.

    Input:
        A string containing biomedical text.

    Output:
        A JSON-like string containing the extracted entities and the input string.

    """
    result = ner.extract_entities(text)

    output = {
        "input_test": text,
        "extracted_entities": result,
    }

    return str(output)


if __name__ == "__main__":
    sample_text = "The patient was diagnosed with diabetes and prescribed metformin."
    output = extract_data.invoke({"text": sample_text})
    print(output)

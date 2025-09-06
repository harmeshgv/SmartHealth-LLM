import spacy

# Load the biomedical NER model
nlp = spacy.load("en_ner_bc5cdr_md")

def extract_symptoms(text: str):
    """
    Extract symptoms/diseases from user input using Spacy biomedical NER.
    Returns a list of symptoms/diseases.
    """
    doc = nlp(text)
    symptoms = [ent.text.lower() for ent in doc.ents if ent.label_ == "DISEASE"]
    return symptoms

print(extract_symptoms("high fever with mild cold"))
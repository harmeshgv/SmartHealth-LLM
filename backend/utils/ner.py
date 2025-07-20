import spacy


class NER:
    def __init__(self):
        self.nlp = spacy.load("en_ner_bc5cdr_md")


    def extract_entities(self,text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        doc = self.nlp(text)
        return [ent.text.lower() for ent in doc.ents if ent.label_ == "DISEASE"]

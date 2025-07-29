import re
import spacy

class Text_Preprocessing:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def work_with_extras(self, text: str) -> list:
        text = re.sub(r"[.,:]", " ", text)
        return text.split()

    def work_with_spaces(self, tokens: list) -> list:
        return [word.strip().lower() for word in tokens if word.strip()]

    def process_text(self, tokens: list) -> list:
        doc = self.nlp(" ".join(tokens))
        return [
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop and not token.like_num and not token.is_punct
        ]

    def go_on(self, text) -> str:
        if not isinstance(text, str):  # Safe guard
            text = ""

        tokens = self.work_with_extras(text)
        tokens = self.work_with_spaces(tokens)
        tokens = self.process_text(tokens)
        return " ".join(tokens)  # ✅ Return as string for embedding

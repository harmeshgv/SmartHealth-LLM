import os

BASE_DIR = "app/llm/prompts"

def load_prompt(filename: str) -> str:
    """
    Loads a prompt file from app/llm/prompts folder.
    """
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

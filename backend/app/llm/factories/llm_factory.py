from app.config import settings
from app.llm.adapters.groq_adapter import GroqLLM
from app.llm.adapters.ollama_adapter import OllamaLLM

class LLMFactory:

    PROVIDERS = {
        "groq": GroqLLM,
        "ollama": OllamaLLM,
    }

    @staticmethod
    def for_agent(agent_name: str):
        """
        Use AGENT_MODEL_MAP from settings to load correct LLM.
        """
        provider, model = settings.AGENT_MODEL_MAP[agent_name]
        LLMClass = LLMFactory.PROVIDERS[provider]
        return LLMClass(model)

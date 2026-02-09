from groq import Groq
from anyio.to_thread import run_sync

from app.llm.base.base_llm import BaseLLM
from app.config import settings


class GroqLLM(BaseLLM):

    def __init__(self, model):
        self.model = model
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def generate(self, prompt: str, **kwargs):

        def blocking_call():
            """Groq SDK is synchronous, so run it in a thread."""
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return resp.choices[0].message.content

        return await run_sync(blocking_call)

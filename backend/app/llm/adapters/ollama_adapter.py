import httpx
from app.llm.base.base_llm import BaseLLM


class OllamaLLM(BaseLLM):

    def __init__(self, model="llama3", host="http://localhost:11434"):
        self.model = model
        self.host = host

    async def generate(self, prompt: str, **kwargs):
        async with httpx.AsyncClient(timeout=200) as client:
            # STREAMING REQUEST to Ollama
            resp = await client.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt},
                timeout=None,
            )

            text_output = ""

            # Ollama sends NDJSON lines — read each one
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue

                try:
                    data = httpx.Response(200, content=line).json()
                except Exception:
                    continue

                if "response" in data:
                    text_output += data["response"]

                if data.get("done", False):
                    break

            return text_output.strip()

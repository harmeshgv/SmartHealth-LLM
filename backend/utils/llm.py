from dotenv import load_dotenv
import os
from gravixlayer import GravixLayer

class LLM:  
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GRAVIXLAYER_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ GRAVIXLAYER_API_KEY not found in environment")

        self.client = GravixLayer()
        self.model = "meta-llama/llama-3.1-8b-instruct"

    def talk(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content



class LLM:  
    def __init__(self, api_key=None):
        if api_key:
            self.api_key = api_key
        else:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            self.api_key = os.getenv("GRAVIXLAYER_API_KEY")

        if not self.api_key:
            raise ValueError("⚠️ GRAVIXLAYER_API_KEY not provided!")

        from gravixlayer import GravixLayer
        self.client = GravixLayer(api_key=self.api_key)  # pass key to client
        self.model = "meta-llama/llama-3.1-8b-instruct"

    def talk(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

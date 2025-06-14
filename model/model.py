from crewai import LLM

class GeminiModel:
    def __init__(self, config):
        self.llm = LLM(
            model="gemini/gemini-1.5-flash",  # ✅ Free-tier-friendly
            api_key=config['model']['api_key']
        )

    def get_llm(self):
        return self.llm


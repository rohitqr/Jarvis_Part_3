# openai_llm.py

import os
import openai
from dotenv import load_dotenv
from livekit.agents.llm import LLM

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIModel(LLM):
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature

    async def chat(self, messages: list[dict], **kwargs) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI Error: {e}"

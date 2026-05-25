import requests

from language_models.model import Model
from language_models.utils import add_retries, limiter


class OllamaModel(Model):
    def __init__(self, name, temperature=0.7, base_url="http://192.168.178.167:11435"):
        super().__init__(name)
        self.temperature = temperature
        self.base_url = base_url

    @add_retries
    @limiter.ratelimit('identity', delay=True)
    def generate_response(self, prompt, n_completions=1):
        responses = []

        for _ in range(n_completions):
            res = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "think": False,
                }
            )

            if res.status_code != 200:
                raise Exception(f"Ollama error: {res.text}")

            data = res.json()
            responses.append(data["response"])

        return responses
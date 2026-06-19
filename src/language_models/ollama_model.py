import time

import requests

from language_models.model import Model
from language_models.utils import add_retries, limiter


class OllamaModel(Model):
    def __init__(self, name, temperature=0.7, base_url="http://0.0.0.0:11434"):
        super().__init__(name)
        self.temperature = temperature
        self.base_url = base_url

    @add_retries
    @limiter.ratelimit('identity', delay=True)
    def generate_response(self, prompt, n_completions=1):
        responses = []
        starting_time = time.time()
        for _ in range(n_completions):
            print("Generating response for prompt:\n", prompt)

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

        ending_time = time.time()
        print("Completed generating response in {:.2f} seconds".format(ending_time - starting_time))
        return responses
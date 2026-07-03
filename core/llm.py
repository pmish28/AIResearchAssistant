import requests as request

class LLM:
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name

    def generate(self, prompt):
        response = request.post(f"http://localhost:11434/api/generate", 
                                json={"model": self.model_name, "prompt": prompt, "stream": False})
        if response.status_code == 200:
            return response.json().get("response", "")


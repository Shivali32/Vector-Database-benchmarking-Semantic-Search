from transformers import pipeline
from openai import OpenAI
import os
import requests

class RAGPipeline:
    def __init__(self, model_name="llama-4-scout"):
        self.api_url = "https://reallms.rescloud.iu.edu/direct/v1/chat/completions"
        self.api_key = os.getenv("API_KEY")
        self.model = model_name
    # def __init__(self, model_name="google/flan-t5-large"):
    #     self.generator = pipeline("text2text-generation", model=model_name)

    def build_prompt(self, query, chunks):
        context = "\n".join(chunks)

        prompt = f"""
        You are a question answering system.

        Answer the question using ONLY the context below and rephrase it to make a meaningful 2-3 sentence paragraph.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        return prompt


    def generate_answer(self, query, retrieved_chunks):
        prompt = self.build_prompt(query, retrieved_chunks)

        response = requests.post(
            self.api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 300
            },
        )

        if response.status_code != 200:
            print("API Error:", response.text)
            return "Error generating answer"

        return response.json()["choices"][0]["message"]["content"]


    # def generate_answer(self, query, retrieved_chunks):
    #     prompt = self.build_prompt(query, retrieved_chunks)

    #     result = self.generator(
    #         prompt,
    #         max_length=200,
    #         do_sample=False
    #     )

    #     return result[0]["generated_text"]
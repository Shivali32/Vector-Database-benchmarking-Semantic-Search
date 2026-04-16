from transformers import pipeline

class RAGPipeline:
    def __init__(self, model_name="google/flan-t5-large"):
        self.generator = pipeline("text2text-generation", model=model_name)

    def build_prompt(self, query, chunks):
        context = "\n".join(chunks)

        prompt = f"""
        You are a question answering system.

        Answer the question using ONLY the context below and rephrase it to make a meaningful sentence.
        


        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        return prompt

    def generate_answer(self, query, retrieved_chunks):
        prompt = self.build_prompt(query, retrieved_chunks)

        result = self.generator(
            prompt,
            max_length=200,
            do_sample=False
        )

        return result[0]["generated_text"]
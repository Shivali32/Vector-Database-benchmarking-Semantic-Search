from transformers import pipeline

class RAGPipeline:
    def __init__(self, model_name="google/flan-t5-base"):
        self.generator = pipeline("text2text-generation", model=model_name)

    def build_prompt(self, query, chunks):
        context = "\n".join(chunks)

        prompt = f"""
        Use the context below to answer the question clearly.

        Context:
        {context}

        Question:
        {query}

        Answer in 2-3 sentences:
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
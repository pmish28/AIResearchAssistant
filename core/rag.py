
class RAGPipeline:
    def __init__(self,  embedder, llm, index, chunks):
        self.embedder = embedder
        self.llm = llm
        self.index = index
        self.chunks = chunks

    def build_prompt(self, context_chunks, question):
        context_text = "\n".join(context_chunks)
        prompt = f""" You are a helpful AI assistant. Use the following context to answer the question. 
        RULES:
            - Use ONLY the context below
            - If the answer is not in the context, say: "I don't know"
            - Do NOT use outside knowledge
        Context = {context_text}
        Question = {question}
        
        Answer clearly and concisely based on the context provided. 
    """
        return prompt

    def answer_question(self, query):
        query_embedding = self.embedder.embed_query(query)  # Embed the query
        distances, indices = self.index.search(query_embedding, 5)
        results = indices[0]
        retrieved_chunks = [self.chunks[i] for i in results]
        prompt = self.build_prompt(retrieved_chunks, query)
        response = self.llm.generate(prompt)
        return response

class Agent:
    def __init__(self, rag_pipeline, llm, memory):
        self.rag =rag_pipeline
        self.llm = llm
        self.memory = memory

    def decide_action(self, query):
        
        """ Simple rule based decision-making
        """
        keywords = ["pdf", "document", "notes", "text", "according to", "file"]

        if any(keyword in query.lower() for keyword in keywords):
            return "rag"
        return "llm"
    
    def run(self, query):
        self.memory.store("user", query)
        print("Inside run")

        mode = self.decide_action(query)
        if mode == "rag":
            response = self.rag.answer_question(query)  
        else:
            prompt = f"""You are a helpful AI assistant. Answer the following question: {query}
            chat_history = {self.memory.get_history()} user: {query} assistant: """
            
            response = self.llm.generate(prompt)
            print("response inside agent run ",response)

            self.memory.store("assistant", response)
           
        return response
from pathlib import Path
from pypdf import PdfReader
from core.agent import Agent
from sklearn import pipeline
from core.chunker import chunk_text
from core.document_loader import load_pdf
from core.llm import LLM
from core.memory import Memory
from core.rag import RAGPipeline
from core.embeddings import EmbeddingService


if __name__ == "__main__":

    pdf_path = Path("data/Sample_AI_Notes.pdf")    
    pdf_text = load_pdf(str(pdf_path))
    chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)

    embedder =EmbeddingService() 
    llm = LLM()

    chunk_embeddings = embedder.embed_texts(chunks)
    index = embedder.build_faiss_index(chunk_embeddings)  
    
    rag = RAGPipeline(embedder, llm, index, chunks)

    memory = Memory()
    agent = Agent(rag, llm, memory)

    while True: 
        query = input("You:")
        response = agent.run(query)
        print(f"AI agent run generated this answer: {response} ")
        

from sentence_transformers import SentenceTransformer
import faiss    

  
class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)  # This converts every text chunk into a numerical vector.

    def build_faiss_index(self, embeddings):  # This creates an in-memory vector database.
        dimension = embeddings.shape[1]  # Get the dimension of the embeddings
        index = faiss.IndexFlatL2(dimension)  # Create a FAISS index for L2 distance
        index.add(embeddings)  # Add the embeddings to the index
        return index

    def search_index(self, index, query_embedding, top_k=5):  # This searches the vector database for the 5 most similar chunks to the query.
        distances, indices = index.search(query_embedding, top_k)  # Search the index
        return indices[0]

    def embed_query(self, query):
        return self.model.encode([query], convert_to_numpy=True)  # This converts the query into a numerical vector.
    
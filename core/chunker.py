
def chunk_text(text, chunk_size=1000, overlap=200):
    
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move the start index forward by chunk_size - overlap
        start += chunk_size - overlap

    return chunks
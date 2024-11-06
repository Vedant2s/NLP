import google.generativeai as genai

API_KEY="Your api"
genai.configure(api_key=API_KEY)

class GoogleEmbeddingWrapper:
    def embed_documents(self, texts):
        # Generate embeddings for a list of texts
        embeddings = []
        for text in texts:
            embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                title="Embedding of single string"
            )
            embeddings.append(embedding['embedding'])  # Ensure this is the correct field for the embedding vector
        return embeddings

    def embed_query(self, text):
        # Generate an embedding for a single query text
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Embedding of single string"
        )
        return embedding['embedding']  # Ensure this is the correct field for the embedding vector

def get_embedding_function():
    return GoogleEmbeddingWrapper()

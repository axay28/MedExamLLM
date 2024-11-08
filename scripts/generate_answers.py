import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Initialize the embedding model and ChromaDB client
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# Initialize the Hugging Face model pipeline for text generation
generator = pipeline("text-generation", model="gpt2")

# Define the collection
collection_name = "vector-store-rare-diseases"
collection = chroma_client.get_collection(collection_name)

# RAG function
def rag_query(query):
    print("Embedding the query...")
    query_embedding = embedding_model.encode([query])[0]

    print("Querying ChromaDB...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas"]
    )

    print("Preparing context from retrieved documents...")
    context = "\n".join(results['documents'][0])
    print("Retrieved context from ChromaDB:", context)


    print("Generating response using Hugging Face model...")
    generated_text = generator(
        f"Context: {context}\nQuestion: {query}\nAnswer:",
        max_new_tokens=50,  # Specify how many new tokens to generate
        num_return_sequences=1,
        do_sample=True
    )[0]['generated_text']

    return generated_text

# Example query
query = "What are the symptoms of Prader-Willi Syndrome?"
answer = rag_query(query)
print("Final Answer:", answer)

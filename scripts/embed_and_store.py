# embed_and_store.py

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client and Sentence-BERT model
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define or connect to the ChromaDB collection
collection_name = 'vector-store-rare-diseases'
collection = chroma_client.get_or_create_collection(collection_name)

# Path to the folder containing processed chunk CSV files
chunks_directory = "/Users/akshaymulgund/Projects/PubMedScraper/chunks"

# Function to embed and store chunks in ChromaDB
def store_chunks_in_chromadb(file_path):
    # Load the chunked data from the CSV file
    chunks_df = pd.read_csv(file_path)

    # Process each row (chunk) in the DataFrame
    for _, row in chunks_df.iterrows():
        # Generate embedding for the text chunk
        embedding = model.encode(row['text'])
        
        # Define metadata
        metadata = {
            "Title": row["title"] if pd.notnull(row["title"]) else "",
            "Publication Date": row["publication_date"] if pd.notnull(row["publication_date"]) else "",
            "Authors": row["authors"] if pd.notnull(row["authors"]) else "",
            "Disease": row["disease"]
        }
        
        # Add the embedding to the ChromaDB collection
        collection.add(
            documents=[row['text']],
            metadatas=[metadata],
            ids=[row['chunk_id']],
            embeddings=[embedding]
        )

    print(f"Stored chunks from {file_path} into ChromaDB.")

# Loop through each CSV file in the chunks directory and process it
for filename in os.listdir(chunks_directory):
    if filename.endswith(".csv"):
        # Construct the full file path
        file_path = os.path.join(chunks_directory, filename)
        
        # Store the chunks in ChromaDB
        store_chunks_in_chromadb(file_path)


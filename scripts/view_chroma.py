import chromadb

# Connect to ChromaDB running on Docker
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# List all collections
print("Collections in ChromaDB:")
collections = chroma_client.list_collections()
for collection in collections:
    print(f"- Collection Name: {collection.name}")

# Select and view contents of a specific collection
collection_name = "vector-store-rare-diseases"  # Change to your collection's name
try:
    collection = chroma_client.get_collection(collection_name)
    print(f"\nInspecting collection: {collection_name}")

    # Retrieve all items but limit the display to a subset (e.g., 5 items)
    results = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    # Display a subset of the retrieved items (e.g., the first 5)
    for document, metadata, embedding in zip(results['documents'][:5], results['metadatas'][:5], results['embeddings'][:5]):
        print("Document:", document)
        print("Metadata:", metadata)
        print("Embedding:", embedding)  # Embedding will be a long list of numbers
        print("------")

except Exception as e:
    print(f"Error retrieving collection '{collection_name}':", str(e))

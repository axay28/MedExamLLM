# clean_and_chunk.py

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Path to the directory containing your CSV files
input_directory = "/Users/akshaymulgund/Projects/PubMedScraper"
output_directory = os.path.join(input_directory, "chunks")

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize LangChain's TextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,   # Define the size of each chunk
    chunk_overlap=50  # Define overlap between chunks for context continuity
)

# Function to clean and chunk a single CSV file
def clean_and_chunk_csv(file_path, disease_name):
    df = pd.read_csv(file_path)
    processed_chunks = []

    # Process each row (article) in the CSV
    for idx, row in df.iterrows():
        # Start with an empty text variable
        text = ""

        # Add title if available
        if 'Title' in df.columns and pd.notnull(row['Title']):
            text += f"{row['Title']} "

        # Add abstract if available
        if 'Abstract' in df.columns and pd.notnull(row['Abstract']):
            text += f"{row['Abstract']} "

        # Add full text if available
        if 'Full Text' in df.columns and pd.notnull(row['Full Text']):
            text += f"{row['Full Text']}"

        # Clean up text by stripping any extraneous whitespace
        text = text.strip()

        # Split the text into chunks, even if text is empty
        chunks = text_splitter.split_text(text or "No content available.")
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "chunk_id": f"{idx}_{i}",  # Unique ID for each chunk
                "title": row["Title"] if 'Title' in df.columns and pd.notnull(row["Title"]) else "",
                "publication_date": row.get("Publication Date", ""),
                "authors": row.get("Authors", ""),
                "disease": disease_name
            }
            processed_chunks.append(chunk_data)

    # Convert to DataFrame and save as a CSV in the "chunks" folder
    chunks_df = pd.DataFrame(processed_chunks)
    output_path = os.path.join(output_directory, f"processed_{disease_name}_chunks.csv")
    chunks_df.to_csv(output_path, index=False)
    print(f"Chunks saved to {output_path}")

# Loop through each CSV file in the directory and process it
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        # Construct full file path
        file_path = os.path.join(input_directory, filename)
        
        # Extract disease name from filename (assumes format "DiseaseName_articles_2019_2024.csv")
        disease_name = filename.replace("_articles_2019_2024.csv", "")
        
        # Process the CSV file
        clean_and_chunk_csv(file_path, disease_name)

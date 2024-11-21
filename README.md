
# MedExamLLM

**MedExamLLM** is a project that leverages Large Language Models (LLMs) and **ChromaDB embeddings** to create a system for generating medical exam questions related to rare diseases. The project aims to assist medical professionals and geneticists by providing relevant, high-quality questions based on the latest research on rare diseases.

---

## Description

MedExamLLM uses LLMs and ChromaDB to extract insights from medical literature, store semantic embeddings, and generate medical exam answers. This project is especially useful for medical professionals and educators seeking relevant content on rare diseases for training and assessments.

### Key Components:

- **LLM-Based Insights**: Uses LLMs to process and extract insights from medical literature on rare diseases.
- **ChromaDB for Embeddings**: Stores embeddings in ChromaDB, enabling efficient semantic search and relevant information retrieval.
- **Exam Answer Generation**: Automatically generates exam-style answers related to rare diseases, ideal for training and assessment purposes.

### Folder Structure

- **data/**: Contains CSV files with extracted articles on various rare diseases.
- **chunks/**: Processed text chunks from literature to prepare for embedding.
- **scripts/**: Contains Python scripts for different stages of the pipeline:
  - `pubmed_scraper.py`: Scrapes literature on rare diseases from PubMed.
  - `clean_and_chunk.py`: Cleans and chunks literature text for embedding.
  - `embed_and_store.py`: Generates embeddings and stores them in ChromaDB.
  - `generate_answers.py`: Retrieves context from ChromaDB and generates answers using LLMs.
  - `view_chroma.py`: Utility for viewing stored collections and embeddings in ChromaDB.
- **chromadb/**: Dockerized ChromaDB instance with persistent storage.
- **start_chromadb.sh**: Shell script to start ChromaDB server in Docker.

---

## Key Features

- **LLM-Based Insights**: Utilizes models like GPT-4 for extracting key insights from medical literature related to rare diseases.
- **ChromaDB Integration**: Stores and retrieves embeddings, enhancing semantic search and relevance.
- **Exam Question Generation**: Generates exam questions on rare diseases to aid in training medical professionals.

---

## Setup and Requirements

### Prerequisites

- **Python 3.x**
- **Docker** (for running ChromaDB in a container)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/MedExamLLM.git
   cd MedExamLLM
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** for OpenAI API key:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

4. **Start ChromaDB with Docker**:

   Ensure Docker is running, then execute:

   ```bash
   ./start_chromadb.sh
   ```

   This will launch a ChromaDB instance on `localhost:8000`.

### Running the Pipeline

1. **Scrape Articles from PubMed**:

   Run `pubmed_scraper.py` to fetch rare disease literature from PubMed and save it to CSVs in the `data/` folder.

   ```bash
   python scripts/pubmed_scraper.py
   ```

2. **Clean and Chunk Data**:

   Prepare text data by cleaning and chunking it for embedding.

   ```bash
   python scripts/clean_and_chunk.py
   ```

3. **Embed and Store in ChromaDB**:

   Generate embeddings for each chunk and store them in ChromaDB for efficient retrieval.

   ```bash
   python scripts/embed_and_store.py
   ```

4. **Generate Answers Using LLM and ChromaDB Context**:

   Retrieve relevant chunks from ChromaDB based on queries, and use an LLM to generate answers.

   ```bash
   python scripts/generate_answers.py
   ```

5. **Inspect ChromaDB** (Optional):

   Use `view_chroma.py` to list collections and verify stored embeddings in ChromaDB.

   ```bash
   python scripts/view_chroma.py
   ```

---

## Usage Examples

To ask a question and receive an answer based on rare disease literature:

1. **Start ChromaDB** (if not already running):

   ```bash
   ./start_chromadb.sh
   ```

2. **Run `generate_answers.py` with a query**:

   ```bash
   python scripts/generate_answers.py
   ```

This will retrieve context from ChromaDB and generate an answer using the LLM based on stored embeddings.

---

## Additional Notes

- **Data Storage**: Literature and processed data are stored in the `data/` and `chunks/` folders.
- **Embedding Storage**: ChromaDB is configured to store embeddings persistently in a Docker volume to ensure data remains across sessions.
- **Customizability**: You can modify scripts or the ChromaDB configuration to adapt to other data sources or embedding methods.

---

## Troubleshooting

- If ChromaDB is not responding, ensure Docker is running and restart it with start_chromadb.sh.
- For issues with the Hugging Face model, ensure that the model is available locally or configured properly in your environment.
- Use view_chroma.py to inspect collections and verify that embeddings are stored correctly in ChromaDB.




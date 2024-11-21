# MedExamLLM

**MedExamLLM** is a project that leverages Retrieval-Augmented Generation (RAG) using Large Language Models (LLMs) and ChromaDB embeddings to generate high-quality, contextually accurate medical exam questions focused on rare diseases. By combining the power of LLMs with the retrieval of relevant medical literature, the system dynamically generates questions and answers based on the most up-to-date information. This approach allows medical professionals, educators, and researchers to access relevant content for training, assessments, and clinical decision-making in rare disease contexts.


---

## Description

MedExamLLM uses LLMs and ChromaDB to extract insights from medical literature, store semantic embeddings, and generate medical exam answers. This project is especially useful for medical professionals and educators seeking relevant content on rare diseases for training and assessments.

### Key Components:

- **LLM-Based Insights**: Uses LLMs to process and extract insights from medical literature on rare diseases.
- **ChromaDB for Embeddings**: Stores embeddings in ChromaDB, enabling efficient semantic search and relevant information retrieval.
- **Exam Answer Generation**: Automatically generates exam-style answers related to rare diseases, ideal for training and assessment purposes.

### Folder Structure

- **data/**: Contains CSV files with extracted articles on various rare diseases.
- **articles/**: Folder where the articles scraped from PubMed are stored.
- **chunks/**: Processed text chunks from literature to prepare for embedding.
- **scripts/**: Contains Python scripts for different stages of the pipeline:
  - `pubmed_scraper.py`: Scrapes literature on rare diseases from PubMed.
  - `clean_and_chunk.py`: Cleans and chunks literature text for embedding.
  - `embed_and_store.py`: Generates embeddings and stores them in ChromaDB.
  - `generate_answers.py`: Retrieves context from ChromaDB and generates answers using LLMs.
  - `view_chroma.py`: Utility for viewing stored collections and embeddings in ChromaDB.
  - `benchmarkclaude.py`: Benchmarks and evaluates the performance of Claude.
  - `benchmarkllama.py`: Benchmarks and evaluates the performance of Llama.
  - `evaluate_llama.py`: Evaluates the results from the Llama benchmark.
  - `filter.py`: Filters the dataset for relevant rare diseases.
  - `evaluate_accuracy.py`: Evaluates the accuracy of model predictions.
- **chromadb/**: Dockerized ChromaDB instance with persistent storage.
- **start_chromadb.sh**: Shell script to start ChromaDB server in Docker.

---

## Key Features

- **LLM-Based Insights**: Utilizes models like Claude and Llama for extracting key insights from medical literature related to rare diseases.
- **ChromaDB Integration**: Stores and retrieves embeddings, enhancing semantic search and relevance.
- **Benchmarking**: Evaluate model performance using scripts like `benchmarkclaude.py` and `benchmarkllama.py`.
- **Exam Question Generation**: Generates exam questions on rare diseases to aid in training medical professionals.

---

## Setup and Requirements

### Prerequisites

- **Python 3.x**: Ensure Python 3 is installed on your machine. You can verify by running `python3 --version`.
- **Docker**: Docker is required to run ChromaDB. You can verify if Docker is installed by running `docker --version`.

Here's the converted GitHub README:

# MedExamLLM

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/MedExamLLM.git
   cd MedExamLLM
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   For OpenAI API:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

   Alternatively, create a `.env` file in the root directory.

4. **Start ChromaDB with Docker**:

   ```bash
   ./start_chromadb.sh
   ```

## Running the Pipeline

### Scrape Articles from PubMed

```bash
python scripts/pubmed_scraper.py
```

### Clean and Chunk Data

```bash
python scripts/clean_and_chunk.py
```

### Generate Embeddings and Store in ChromaDB

```bash
python scripts/embed_and_store.py
```

### Generate Answers Using LLM and ChromaDB Context

```bash
python scripts/generate_answers.py
```

### Benchmark and Evaluate Model Performance

**Benchmark Claude**:

```bash
python scripts/benchmarkclaude.py
```

**Benchmark Llama**:

```bash
python scripts/benchmarkllama.py
```

**Evaluate Results**:

```bash
python scripts/evaluate_accuracy.py
```

### Inspect ChromaDB (Optional)

```bash
python scripts/view_chroma.py
```

## Usage Examples

1. Start ChromaDB:

   ```bash
   ./start_chromadb.sh
   ```

2. Generate answers:

   ```bash
   python scripts/generate_answers.py
   ```

Citations:
[1] https://github.com/your-username/MedExamLLM.git

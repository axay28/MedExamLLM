## Description

**MedExamLLM** leverages Large Language Models (LLMs) and **ChromaDB embeddings** to create a system for generating medical exam questions related to rare diseases. The project aims to assist medical professionals and geneticists by providing relevant and high-quality questions based on the latest research on rare diseases.

### This project includes:

- **LLM-Based Insights**: Uses LLMs for extracting insights from medical literature.
- **ChromaDB Integration**: Stores and retrieves embeddings, improving search and relevance of the information.
- **Medical Exam Generation**: Generates exam-style questions based on rare disease literature to assist in training and assessment.

### Key Features

- **LLM-Based Insights**: Utilizes LLMs (like GPT-4, etc.) to process medical literature and extract key insights related to rare diseases.
- **ChromaDB Integration**: Embeddings are stored and retrieved via ChromaDB for enhanced semantic search and relevance.
- **Medical Exam Generation**: Automatically generates exam questions tailored to rare diseases, helping in assessments for medical exams.

---

## Requirements

- **Python 3.x**
- Required Python Libraries:
  - `biopython`
  - `pandas`
  - `chromadb`
  - `openai`

To install dependencies, run:

```bash
pip install -r requirements.txt

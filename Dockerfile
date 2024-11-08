# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install ChromaDB and Uvicorn
RUN pip install chromadb uvicorn fastapi

# Expose the default port for ChromaDB
EXPOSE 8000

# Command to start the ChromaDB FastAPI server
CMD ["uvicorn", "chromadb.api:app", "--host", "0.0.0.0", "--port", "8000"]

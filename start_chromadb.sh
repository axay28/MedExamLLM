#!/bin/bash
echo "Starting ChromaDB server on localhost:8000"
docker run -p 8000:8000 chromadb/chromadb:latest


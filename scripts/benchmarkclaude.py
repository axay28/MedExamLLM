import anthropic
import os
import json
import requests
import re
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize the ChromaDB client and sentence transformer model
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Set API key and initialize the Anthropic client
API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Replace this with your actual API key if not using environment variables
client = anthropic.Client(api_key=API_KEY)

# Constants for prompts
HUMAN_PROMPT = "\n\nHuman: You are an expert on rare diseases. Please answer the following multiple-choice question by only responding with the number corresponding to the correct answer: 0 for A, 1 for B, 2 for C, or 3 for D."
AI_PROMPT = "\n\nAssistant: Please respond with only the correct option: 0 for A, 1 for B, 2 for C, or 3 for D."


def process_model_response(model_response):
    """Process the model response to extract the answer."""
    match = re.search(r'\(?[a-dA-D]\)?', model_response.strip())
    if match:
        answer = match.group(0).strip("()").upper()  # Remove parentheses and convert to uppercase
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return answer_map.get(answer, None)
    else:
        print(f"Invalid model response: {model_response}")
        return None

def claude_generate(query, context):
    """Generate answers using Claude via Anthropic's Messages API."""
    try:
        # Construct the prompt in the correct format
        prompt = f"{HUMAN_PROMPT} Context: {context}\nQuestion: {query}{AI_PROMPT}"

        # Call the Anthropic API
        response = client.completions.create(
            model="claude-2",  # Use the Claude-2 model
            prompt=prompt,  # Provide the prompt explicitly
            max_tokens_to_sample=300,  # Set the max tokens for the response
            stop_sequences=["\n"],  # Ensure it stops after the answer is given
            stream=False  # Optional: set to False to receive the full response
        )

        # Return the processed response
        return response.completion
    except Exception as e:
        print("Error in Claude generation:", e)
        return "Error in Claude generation."


def retrieve_context(query, db, collection_name="vector-store-rare-diseases", num_results=3):
    """Retrieve relevant context from ChromaDB."""
    try:
        query_embedding = embedder.encode([query])[0]  # Convert query to embedding
        collection = db.get_collection(collection_name)
        results = collection.query(query_embedding, n_results=num_results)

        # Access documents and context
        documents = results['documents'][0]  # Get the first document
        context = "\n".join(documents)
        metadata = results['metadatas'][0]  # Optional metadata
        return context, metadata
    except Exception as e:
        print(f"Error retrieving context from ChromaDB: {e}")
        return "", None

def benchmark_claude(dataset, num_samples=10, db=None):
    results = []

    for idx, data in enumerate(dataset):
        if idx >= num_samples:
            break  # Limit to num_samples

        try:
            question = data.get("question", "")
            choices = [
                data.get("opa", ""),
                data.get("opb", ""),
                data.get("opc", ""),
                data.get("opd", ""),
            ]
            correct_option_index = data.get("cop", None)

            if question and choices and correct_option_index is not None:
                correct_answer = chr(65 + correct_option_index)  # Convert 0-3 to A-D

                # Get context from ChromaDB
                context, metadata = retrieve_context(question, db)

                # Generate the model's response using Claude API
                response = claude_generate(question, context)

                # Process response to extract valid answer
                if response:
                    # Check for a valid answer (A, B, C, D)
                    valid_answers = ['a', 'b', 'c', 'd']
                    response_lower = response.strip().lower()

                    # If response is a valid answer, use it directly
                    if response_lower in valid_answers:
                        model_response = response_lower.upper()
                        results.append({
                            "question": question,
                            "choices": choices,
                            "correct_option": correct_answer,
                            "model_response": model_response,
                            "context": context,
                            "metadata": metadata
                        })
                    else:
                        # If response isn't a direct match, check for keywords or patterns
                        if 'a' in response_lower:
                            model_response = 'A'
                        elif 'b' in response_lower:
                            model_response = 'B'
                        elif 'c' in response_lower:
                            model_response = 'C'
                        elif 'd' in response_lower:
                            model_response = 'D'
                        else:
                            model_response = 'Invalid'

                        if model_response != 'Invalid':
                            results.append({
                                "question": question,
                                "choices": choices,
                                "correct_option": correct_answer,
                                "model_response": model_response,
                                "context": context,
                                "metadata": metadata
                            })
                        else:
                            print(f"Skipping invalid model response at index {idx}: {response}")
                else:
                    print(f"Skipping empty response at index {idx}")

            else:
                print(f"Skipping invalid entry at index {idx}: {data}")
        except Exception as e:
            print(f"Error processing entry at index {idx}: {e}")

    # Save results to file
    with open("claude_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Benchmark completed. Results saved to 'claude_benchmark_results.json'.")


def evaluate_accuracy(results_file):
    """Evaluate accuracy and other metrics."""
    with open(results_file, "r") as f:
        results = json.load(f)

    total = len(results)
    correct = 0
    y_true = []  # List to store the true labels
    y_pred = []  # List to store the predicted labels

    for entry in results:
        expected = entry.get("correct_option", "").strip().lower()
        generated = entry.get("model_response", "").strip().lower()

        # Map expected and generated responses to indices
        answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

        if expected and generated:
            expected_idx = answer_map.get(expected, -1)
            generated_idx = answer_map.get(generated, -1)

            if expected_idx != -1 and generated_idx != -1:  # Valid entries
                y_true.append(expected_idx)
                y_pred.append(generated_idx)

                if expected_idx == generated_idx:
                    correct += 1

    if total == 0:
        print("No valid entries to evaluate.")
        return

    # Calculate accuracy
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if len(y_true) == 0 or len(y_pred) == 0:
        print("No valid comparisons to evaluate precision, recall, and F1.")
        return

    # Precision, Recall, and F1-score
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Precision (macro): {precision * 100:.2f}%")
    print(f"Recall (macro): {recall * 100:.2f}%")
    print(f"F1-score (macro): {f1 * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only plot confusion matrix if there is valid data
    if cm.size > 0:
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["A", "B", "C", "D"], yticklabels=["A", "B", "C", "D"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
    else:
        print("Confusion Matrix is empty. Cannot generate the heatmap.")

if __name__ == "__main__":
    # Load the filtered dataset
    with open("filtered_rare_disease_data.json", "r") as f:
        filtered_dataset = json.load(f)

    # Benchmark Claude with RAG and ChromaDB
    print("Starting Benchmark with Claude...")
    benchmark_claude(filtered_dataset, num_samples=153, db=chroma_client)  # Use the actual number of samples

    # Evaluate Results
    #print("Evaluating Results...")
    #evaluate_accuracy("claude_benchmark_results.json")

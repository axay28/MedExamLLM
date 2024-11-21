import json
import requests
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to ChromaDB running on Docker
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# Function to retrieve relevant context from ChromaDB
def retrieve_context(query, db, collection_name="vector-store-rare-diseases", num_results=3):
    try:
        query_embedding = embedder.encode([query])[0]  # We get the first embedding (for the query)
        collection = db.get_collection(collection_name)
        results = collection.query(query_embedding, n_results=num_results)

        # Access documents correctly from the results
        documents = results['documents'][0]  # First document in the list
        context = "\n".join(documents)

        metadata = results['metadatas'][0]
        return context, metadata
    except Exception as e:
        print(f"Error retrieving context from ChromaDB: {e}")
        return "", None

def process_model_response(model_response):
    """
    Extracts the number (0-3) from the model's response.
    Handles both 'A', 'B', 'C', 'D' and '(A)', '(B)', '(C)', '(D)' formats.
    """
    match = re.search(r'\(?[a-dA-D]\)?', model_response.strip())

    if match:
        answer = match.group(0).strip("()").upper()  # Remove parentheses if present and convert to uppercase
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return answer_map.get(answer, None)  # Map to 0, 1, 2, 3 or return None if invalid
    else:
        print(f"Invalid model response: {model_response}")
        return None

def llama_generate(question, context):
    try:
        payload = {
            "model": "llama2:latest",
            "prompt": f"""You are an expert on rare diseases.
Context: {context}
Question: {question}
Please respond ONLY with the number corresponding to the correct answer: 0 for A, 1 for B, 2 for C, or 3 for D. 
Do not provide any explanation or additional text. Just respond with the number: 0, 1, 2, or 3."""
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        generated_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    generated_text += data.get("response", "")
                except json.JSONDecodeError:
                    pass

        processed_response = process_model_response(generated_text)

        if processed_response is not None:
            return processed_response
        else:
            return None  # Return None for invalid response

    except Exception as e:
        print("Error in LLaMA generation:", e)
        return None  # Return None if there's an error

def benchmark_llama(dataset, num_samples=10, db=None):
    results = []

    for idx, data in enumerate(dataset):
        if idx >= num_samples:
            break  # Limit to num_samples

        try:
            question = data.get("question", "")
            choices = [data.get("opa", ""), data.get("opb", ""), data.get("opc", ""), data.get("opd", "")]
            correct_option_index = data.get("cop", None)

            if question and choices and correct_option_index is not None:
                correct_answer = chr(65 + correct_option_index)  # Convert 0-3 to A-D

                # Get context from ChromaDB
                context, metadata = retrieve_context(question, db)

                # Combine context and question into the prompt
                prompt = f"Context: {context}\nQuestion: {question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

                # Generate the model's response using the LLaMA API
                response = llama_generate(prompt, context="Rare disease benchmark")

                # If response is valid, add to results
                if response is not None:
                    model_response = chr(65 + int(response))  # Convert number to letter (0 -> A, 1 -> B, etc.)
                    results.append({
                        "question": question,
                        "choices": choices,
                        "correct_option": correct_answer,
                        "model_response": model_response,
                        "context": context,
                        "metadata": metadata
                    })
                    print(f"Processed sample {idx + 1}/{num_samples}")
                else:
                    print(f"Skipping invalid model response at index {idx}: {response}")
            else:
                print(f"Skipping invalid entry at index {idx}: {data}")
        except Exception as e:
            print(f"Error processing entry at index {idx}: {e}")

    # Save results to file
    with open("llama_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Benchmark completed. Results saved to 'llama_benchmark_results.json'.")

# def evaluate_accuracy(results_file):
#     with open(results_file, "r") as f:
#         results = json.load(f)

#     total = len(results)
#     correct = 0
#     y_true = []  # List to store the true labels
#     y_pred = []  # List to store the predicted labels

#     for entry in results:
#         expected = entry.get("correct_answer", "").strip().lower()
#         generated = entry.get("model_response", "").strip().lower()

#         # Map expected and generated responses to indices
#         answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        
#         if expected and generated:
#             expected_idx = answer_map.get(expected, -1)
#             generated_idx = answer_map.get(generated, -1)

#             if expected_idx != -1 and generated_idx != -1:  # Valid entries
#                 y_true.append(expected_idx)
#                 y_pred.append(generated_idx)

#                 if expected_idx == generated_idx:
#                     correct += 1

#     if total == 0:
#         print("No valid entries to evaluate.")
#         return

#     # Calculate accuracy
#     accuracy = correct / total
#     print(f"Accuracy: {accuracy * 100:.2f}%")

#     if len(y_true) == 0 or len(y_pred) == 0:
#         print("No valid comparisons to evaluate precision, recall, and F1.")
#         return

#     # Precision, Recall, and F1-score
#     precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

#     print(f"Precision (macro): {precision * 100:.2f}%")
#     print(f"Recall (macro): {recall * 100:.2f}%")
#     print(f"F1-score (macro): {f1 * 100:.2f}%")

#     # Confusion Matrix
#     cm = confusion_matrix(y_true, y_pred)

#     # Only plot confusion matrix if there is valid data
#     if cm.size > 0:
#         # Plot confusion matrix
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["A", "B", "C", "D"], yticklabels=["A", "B", "C", "D"])
#         plt.title("Confusion Matrix")
#         plt.xlabel("Predicted Label")
#         plt.ylabel("True Label")
#         plt.show()
#     else:
#         print("Confusion Matrix is empty. Cannot generate the heatmap.")

if __name__ == "__main__":
    # Load the filtered dataset
    with open("filtered_rare_disease_data.json", "r") as f:
        filtered_dataset = json.load(f)

    # Benchmark LLaMA with RAG and ChromaDB
    print("Starting LLaMA Benchmark with RAG...")
    benchmark_llama(filtered_dataset, num_samples=153, db=chroma_client)  # Use 5 samples for testing

    # Evaluate Results
    print("Evaluating Results...")
    #evaluate_accuracy("llama_benchmark_results.json")

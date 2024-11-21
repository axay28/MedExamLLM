import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_results(results_file):
    with open(results_file, 'r') as f:
        return json.load(f)

def evaluate_metrics(results):
    total = len(results)
    correct = 0
    y_true = []  # List to store the true labels
    y_pred = []  # List to store the predicted labels

    # Define a mapping from letters to numbers for easier comparison
    answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    
    for entry in results:
        expected = entry.get("correct_option", "").strip().lower()
        generated = entry.get("model_response", "").strip().lower()

        # Map expected and generated responses to indices (0-3)
        if expected and generated:
            expected_idx = answer_map.get(expected, -1)
            generated_idx = answer_map.get(generated, -1)

            if expected_idx != -1 and generated_idx != -1:  # Valid entries
                y_true.append(expected_idx)
                y_pred.append(generated_idx)

                if expected_idx == generated_idx:
                    correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Precision, Recall, and F1-score (using macro average)
    if len(y_true) > 0 and len(y_pred) > 0:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print(f"Precision (macro): {precision * 100:.2f}%")
        print(f"Recall (macro): {recall * 100:.2f}%")
        print(f"F1-score (macro): {f1 * 100:.2f}%")
    else:
        print("Insufficient valid data for precision, recall, and F1 calculation.")

    # Confusion Matrix
    if len(y_true) > 0 and len(y_pred) > 0:
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["A", "B", "C", "D"], yticklabels=["A", "B", "C", "D"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
    else:
        print("Confusion Matrix is empty. No valid comparisons to generate the heatmap.")

if __name__ == "__main__":
    # Path to the file containing the model benchmark results
    results_file = "claude_benchmark_results.json"  # Change this to the path of your results file
    
    # Load the results
    results = load_results(results_file)
    
    # Evaluate metrics
    evaluate_metrics(results)


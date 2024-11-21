import argparse
import openai
import requests
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json
import os
import warnings
warnings.filterwarnings("ignore", message="NotOpenSSLWarning")

# GPT-4 Setup
openai.api_key = ""

# Claude Setup

client = Anthropic(api_key="ANTHROPIC-API-KEY")



def gpt4_generate(query, context):
    """
    Generate answers using GPT-4o via OpenAI API.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a medical expert specializing in rare diseases."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("Error in GPT-4o generation:", e)
        return "Error in GPT-4o generation."


def llama_generate(query, context):
    """
    Generate answers using the LLaMA model via Ollama's local API.
    Handles streaming responses properly.
    """
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama2:latest",
            "prompt": f"""You are an expert on rare diseases.
Context: {context}
Question: {query}
Answer:"""
        }
        response = requests.post(url, json=payload, stream=True)

        # Raise an error for bad HTTP responses
        response.raise_for_status()

        # Collect the streamed chunks
        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    # Parse each line as JSON
                    data = json.loads(line)
                    full_response += data.get("response", "")
                except json.JSONDecodeError:
                    # Skip any malformed lines
                    pass

        return full_response.strip()
    except Exception as e:
        print("Error in LLaMA (Ollama) generation:", e)
        return "Error in LLaMA (Ollama) generation."




# Initialize the Anthropic client with your API key

def claude_generate(query, context):
    """
    Generate answers using Claude via Anthropic's Messages API.
    """
    try:
        # Construct the prompt
        prompt = f"{HUMAN_PROMPT}Context: {context}\nQuestion: {query}{AI_PROMPT}"

        # Call the Anthropic API
        response = client.completions.create(
            model="claude-2",  # Replace with the correct model version
            prompt=prompt,  # Provide the prompt explicitly
            max_tokens_to_sample=300,  # Ensure this parameter is set
            stream=False  # Optional: set to True for streaming
        )

        # Return the response completion
        return response.completion
    except Exception as e:
        print("Error in Claude generation:", e)
        return "Error in Claude generation."




def generate_answer(query, context, model_choice="gpt4"):
    """
    Main function to select the model and generate the answer.
    """
    if model_choice == "gpt4":
        return gpt4_generate(query, context)
    elif model_choice == "llama":
        return llama_generate(query, context)
    elif model_choice == "claude":
        return claude_generate(query, context)
    else:
        return "Invalid model choice. Please select 'gpt4', 'llama', or 'claude'."


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate answers using different models.")
    parser.add_argument("--llama", action="store_true", help="Use LLaMA model via Ollama")
    parser.add_argument("--gpt4", action="store_true", help="Use GPT-4 model via OpenAI")
    parser.add_argument("--claude", action="store_true", help="Use Claude model via Anthropic")

    args = parser.parse_args()

    # Default question and context
    query = "What are the symptoms of Ehlers-Danlos Syndrome?"
    context = "Ehlers-Danlos Syndrome (EDS) is a group of disorders that affect connective tissues, including the skin, joints, and blood vessel walls."

    # Select model based on command-line arguments
    if args.llama:
        print("\nUsing LLaMA (Ollama):")
        print(generate_answer(query, context, model_choice="llama"))
    elif args.gpt4:
        print("\nUsing GPT-4:")
        print(generate_answer(query, context, model_choice="gpt4"))
    elif args.claude:
        print("\nUsing Claude:")
        print(generate_answer(query, context, model_choice="claude"))
    else:
        print("No model specified. Use --llama, --gpt4, or --claude to specify a model.")

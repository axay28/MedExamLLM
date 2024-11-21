import json
from datasets import load_dataset

def split_disease_terms(disease):
    """
    Split each disease name into multiple components for broader matching.
    For example, 'Sickle Cell Anemia' becomes ['Sickle', 'Cell', 'Anemia']
    """
    return disease.split()  # Simple split by space, adjust as needed for other cases

def filter_dataset_with_terms(dataset, disease_list):
    """
    Filter the dataset to include only questions related to the specified diseases or their components.
    """
    filtered_data = []
    # Convert all disease names to lowercase for case-insensitive matching
    disease_list_lower = [disease.lower() for disease in disease_list]
    
    # Generate additional search terms by splitting the disease names
    disease_terms = []
    for disease in disease_list_lower:
        disease_terms.extend(split_disease_terms(disease))  # Split into components

    for data in dataset:
        diseases = data.get("rare disease", [])

        # Ensure that the diseases field is a list and check if any of the disease components are in the disease list
        if isinstance(diseases, list):
            # Check for exact match with any disease or its components
            matching_diseases = [disease for disease in diseases if any(term in disease.lower() for term in disease_terms)]
            if matching_diseases:
                filtered_data.append(data)

    print(f"Filtered dataset contains {len(filtered_data)} entries out of {len(dataset)}.")
    return filtered_data

def save_filtered_data_to_json(filtered_data, filename="filtered_rare_disease_data.json"):
    """
    Save the filtered dataset to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered dataset saved to {filename}.")

# Example of using the filtering function
disease_list = [
    "Cystic Fibrosis", "Huntington Disease", "Marfan Syndrome", "Sickle Cell Anemia",
    "Hemophilia", "Gaucher Disease", "Phenylketonuria", "Tay-Sachs Disease",
    "Alpha-1 Antitrypsin Deficiency", "Duchenne Muscular Dystrophy", "Amyotrophic Lateral Sclerosis",
    "Fabry Disease", "Pompe Disease", "Wilson Disease", "Spinal Muscular Atrophy",
    "Thalassemia", "Neurofibromatosis Type 1", "Hereditary Angioedema", "X-linked Adrenoleukodystrophy",
    "Ehlers-Danlos Syndrome", "Alport Syndrome", "Friedreich Ataxia", "Rett Syndrome",
    "Prader-Willi Syndrome", "Usher Syndrome", "Von Hippel-Lindau Disease", "Tuberous Sclerosis",
    "Batten Disease", "Krabbe Disease", "Leigh Syndrome"
]

# Example dataset loading
redisqa_dataset = load_dataset("guan-wang/ReDis-QA")
filtered_dataset = filter_dataset_with_terms(redisqa_dataset["test"], disease_list)

# Save filtered data to JSON file
save_filtered_data_to_json(filtered_dataset, "filtered_rare_disease_data.json")

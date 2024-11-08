import os
from Bio import Entrez
import pandas as pd
import time
from urllib.error import HTTPError
from Bio.Entrez.Parser import ListElement

# Set your email and API key here
Entrez.email = "mulgunda@oregonstate.edu"  
Entrez.api_key = os.getenv("NCBI_API_KEY")

# Helper function to safely extract data from complex structures
def safe_extract(data, *keys, default=""):
    """
    Safely extracts a nested value from dictionaries or lists.
    Converts ListElement to list if necessary.
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) or isinstance(data, ListElement):
            data = data[0] if len(data) > 0 else default
        else:
            return default
    return data if data else default

# Helper function to flatten nested lists into a string
def flatten_to_string(item):
    """
    Recursively converts nested lists into a single string.
    """
    if isinstance(item, list) or isinstance(item, ListElement):
        return " ".join(flatten_to_string(sub_item) for sub_item in item)
    elif isinstance(item, str):
        return item
    return str(item)

# Function to search PMC (PubMed Central)
def search_pmc(term, start_year, end_year, retries=3):
    query = f"{term}[Title/Abstract] AND {start_year}:{end_year}[dp]"
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt+1} for query: {query}")
            handle = Entrez.esearch(db="pmc", term=query, retmax=30, usehistory="y")
            results = Entrez.read(handle)
            handle.close()
            return results
        except HTTPError as e:
            if e.code == 500:
                print(f"Server error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"An error occurred: {e}")
                break
    return None

# Function to fetch and parse full article details from UIDs in PMC
def fetch_full_text_pmc(id_list):
    ids = ",".join(id_list)
    try:
        handle = Entrez.efetch(db="pmc", id=ids, rettype="full", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        return records
    except Exception as e:
        print(f"Error fetching full-text articles: {e}")
        return None

# Main function to scrape PMC for a disease
def scrape_pmc(disease, start_year="2019", end_year="2024"):
    # Search for articles on a rare disease in PMC
    results = search_pmc(disease, start_year, end_year)
    
    if results is None:
        print("Error fetching data from PMC. No results were returned.")
        return None

    article_ids = results["IdList"]
    
    # Fetch article details if IDs are available
    if len(article_ids) == 0:
        print(f"No articles found for {disease}.")
        return None
    
    articles = fetch_full_text_pmc(article_ids)
    
    data = []
    if isinstance(articles, list):  # Ensures we have multiple articles in list format
        for article in articles:
            try:
                # Extract Title
                title = flatten_to_string(safe_extract(article, "front", "article-meta", "title-group", "article-title", default="No Title"))

                # Extract Abstract
                abstract_parts = safe_extract(article, "front", "article-meta", "abstract", default=[])
                abstract = " ".join([flatten_to_string(part.get("p", "")) for part in abstract_parts if isinstance(part, dict)])

                # Extract Full Text
                full_text_parts = safe_extract(article, "body", "sec", default=[])
                full_text = " ".join([flatten_to_string(section.get("p", "")) for section in full_text_parts if isinstance(section, dict)])

                # Extract Authors
                authors = []
                author_list = safe_extract(article, "front", "article-meta", "contrib-group", default=[])
                if isinstance(author_list, list):
                    for author in author_list:
                        if isinstance(author, dict):
                            surname = flatten_to_string(safe_extract(author, "name", "surname", default=""))
                            given_names = flatten_to_string(safe_extract(author, "name", "given-names", default=""))
                            authors.append(f"{given_names} {surname}".strip())
                authors_str = ", ".join(authors) if authors else "N/A"

                # Extract Publication Date
                pub_date = flatten_to_string(safe_extract(article, "front", "article-meta", "pub-date", "year", default="N/A"))

                data.append({
                    "Title": title,
                    "Abstract": abstract,
                    "Full Text": full_text if full_text else "Full text not available in PMC",
                    "Authors": authors_str,
                    "Publication Date": pub_date
                })

            except Exception as e:
                print(f"Error processing article data: {e}")
    
    # Convert to DataFrame and return
    df = pd.DataFrame(data)
    return df if not df.empty else None

# List of 30 most prevalent rare diseases
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

# Scrape PMC for each disease and save the results
for disease in disease_list:
    print(f"Fetching data for {disease}...")
    df = scrape_pmc(disease)
    if df is not None:
        filename = f"{disease.replace(' ', '_')}_articles_2019_2024.csv"
        df.to_csv(filename, index=False)
        print(f"Data for {disease} saved to {filename}")
    else:
        print(f"No data available for {disease}.")

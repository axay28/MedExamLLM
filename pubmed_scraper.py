import os
from Bio import Entrez
import pandas as pd
import time
from urllib.error import HTTPError

# Set your email and API key here
Entrez.email = "mulgunda@oregonstate.edu"  
Entrez.api_key = os.getenv("NCBI_API_KEY")

# Function to search PubMed
def search_pubmed(term, start_year, end_year, retries=3):
    query = f"{term}[Title/Abstract] AND {start_year}:{end_year}[dp]"
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt+1} for query: {query}")
            handle = Entrez.esearch(db="pubmed", term=query, retmax=30, usehistory="y")
            results = Entrez.read(handle)
            handle.close()
            return results
        except HTTPError as e:
            if e.code == 500:
                print(f"Server error: {e}. Retrying in 5 seconds...")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print(f"An error occurred: {e}")
                break
    return None

# Function to fetch article details from UIDs
def fetch_details(id_list):
    ids = ",".join(id_list)
    try:
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        return records
    except Exception as e:
        print(f"Error fetching article details: {e}")
        return None

# Main function to scrape PubMed for a disease
# Main function to scrape PubMed for a disease
def scrape_pubmed(disease, start_year="2019", end_year="2024"):
    # Search for articles on a rare disease
    results = search_pubmed(disease, start_year, end_year)
    
    if results is None:
        print("Error fetching data from PubMed. No results were returned.")
        return None

    article_ids = results["IdList"]
    
    # Fetch article details if IDs are available
    if len(article_ids) == 0:
        print(f"No articles found for {disease}.")
        return None
    
    articles = fetch_details(article_ids)
    
    # Extract relevant details from the articles
    data = []
    for article in articles["PubmedArticle"]:
        medline_citation = article["MedlineCitation"]
        article_data = medline_citation["Article"]
        title = article_data["ArticleTitle"]
        abstract = article_data.get("Abstract", {}).get("AbstractText", [""])[0]
        
        # Author extraction with error handling
        authors = []
        for author in article_data.get("AuthorList", []):
            last_name = author.get("LastName", "")
            first_name = author.get("ForeName", "")
            if last_name or first_name:
                authors.append(f"{first_name} {last_name}".strip())
        authors_str = ", ".join(authors) if authors else "N/A"
        
        pub_date = article_data["Journal"]["JournalIssue"]["PubDate"].get("Year", "N/A")
        
        data.append({
            "Title": title,
            "Abstract": abstract,
            "Authors": authors_str,
            "Publication Date": pub_date
        })

    # Convert to DataFrame and return
    df = pd.DataFrame(data)
    return df

# List of 30 most prevalent rare diseases (replace with actual diseases)
disease_list = [
    "Cystic Fibrosis",
    "Huntington Disease",
    "Marfan Syndrome",
    "Sickle Cell Anemia",
    "Hemophilia",
    "Gaucher Disease",
    "Phenylketonuria",
    "Tay-Sachs Disease",
    "Alpha-1 Antitrypsin Deficiency",
    "Duchenne Muscular Dystrophy",
    "Amyotrophic Lateral Sclerosis",
    "Fabry Disease",
    "Pompe Disease",
    "Wilson Disease",
    "Spinal Muscular Atrophy",
    "Thalassemia",
    "Neurofibromatosis Type 1",
    "Hereditary Angioedema",
    "X-linked Adrenoleukodystrophy",
    "Ehlers-Danlos Syndrome",
    "Alport Syndrome",
    "Friedreich Ataxia",
    "Rett Syndrome",
    "Prader-Willi Syndrome",
    "Usher Syndrome",
    "Von Hippel-Lindau Disease",
    "Tuberous Sclerosis",
    "Batten Disease",
    "Krabbe Disease",
    "Leigh Syndrome"
]

# Scrape PubMed for each disease and save the results
for disease in disease_list:
    print(f"Fetching data for {disease}...")
    df = scrape_pubmed(disease)
    if df is not None:
        filename = f"{disease.replace(' ', '_')}_articles_2019_2024.csv"
        df.to_csv(filename, index=False)
        print(f"Data for {disease} saved to {filename}")
    else:
        print(f"No data available for {disease}.")

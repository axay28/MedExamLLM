import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Base URL for Orphanet rare disease list
base_url = "https://www.orpha.net/consor/cgi-bin/Disease_Search_List.php?lng=EN&search=Disease_Search_List_Disease&data_id=&data_type=ORPHA&list_type=&display_type=&sort=&subCat=&cat="

# Function to scrape Orphanet rare diseases for each letter (A-Z)
def scrape_orphanet_rare_diseases():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    all_diseases = []

    # Loop through each letter's page
    for letter in letters:
        url = f"{base_url}{letter}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve Orphanet page for letter {letter}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Scraping the disease names listed under the results section
        disease_elements = soup.select('div#list_by_letter ul li a')  # Assuming disease names are in <ul><li><a> format
        
        for disease_element in disease_elements:
            disease_name = disease_element.text.strip()
            if disease_name:
                all_diseases.append(disease_name)

        print(f"Scraped {len(disease_elements)} diseases for letter {letter}")
        time.sleep(1)  # Pause between requests to avoid overwhelming the server

    return list(set(all_diseases))  # Return unique disease names

# Scrape rare diseases from Orphanet and return the list
orphanet_diseases = scrape_orphanet_rare_diseases()

if orphanet_diseases:
    print(f"Scraped {len(orphanet_diseases)} rare diseases from Orphanet.")
    # Convert the list into a DataFrame and save it to CSV for future use
    rare_diseases_df = pd.DataFrame(orphanet_diseases, columns=["Rare Disease"])
    rare_diseases_df.to_csv("orphanet_rare_diseases.csv", index=False)
    print("Orphanet rare diseases saved to 'orphanet_rare_diseases.csv'.")

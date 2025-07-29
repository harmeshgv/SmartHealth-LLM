import requests
from bs4 import BeautifulSoup
import time
import os
import csv
import pandas as pd
from tqdm import tqdm
import sys
import string
from IPython.display import clear_output

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from backend.config import MAYO_CSV

class main:
    def __init__(self):
        self.headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

        self.retries = 3
        self.delay = 5
        
        
    def data_extractor(self, base_url):
        diagnosis_treatment_link = ""
        doctors_departments_link= ""
        for attempt in range(self.retries):
            try:
                response = requests.get(base_url, headers=self.headers, timeout=20)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                content1 = soup.find('a', id="et_genericNavigation_diagnosis-treatment")
                
                if not content1:
                    for a in soup.find_all('a'):
                        link_text = a.get_text(separator=' ').strip().lower()
                        if "diagnosis" in link_text and "treatment" in link_text:
                            content1 = a
                            break
                
                if content1:
                    href1 = content1.get('href')
                    diagnosis_treatment_link = f"https://www.mayoclinic.org{href1}" if href1 and href1.startswith("/") else href1
                content2 = soup.find('a', id="et_genericNavigation_doctors-departments")
                if not content2:
                # fallback: search by link text containing both words
                    for a in soup.find_all('a'):
                        link_text = a.get_text(separator=' ').strip().lower()
                        if "doctors" in link_text and "departments" in link_text:
                            content2 = a
                            break
                if content2:
                    href2 = content2.get('href')
                    doctors_departments_link = f"https://www.mayoclinic.org{href2}" if href2 and href2.startswith("/") else href2

                break  # success, exit retry loop

            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt + 1}] Error fetching {base_url}: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.delay)

        return diagnosis_treatment_link, doctors_departments_link
                      
        


    def web_scraping(self,base_url):
    # Define the expected headers in order
        expected_headers = ["disease", "main_link", "Diagnosis_treatment_link", "Doctors_departments_link"]
    
    # Check if file exists and read existing headers if it does
        file_exists = os.path.isfile(MAYO_CSV)
        existing_headers = []
    
        if file_exists:
            with open(MAYO_CSV, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                existing_headers = next(reader, [])
    
    # Determine if we need to write headers
        write_headers = not file_exists or existing_headers != expected_headers
    
    # Get the webpage content
        response = requests.get(base_url)
        if response.status_code != 200:
            print("Failed to retrieve page")
            exit()

        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.select(".cmp-results-with-primary-name__see-link, .cmp-results-with-primary-name a")

        with open(MAYO_CSV, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
        
        # Write headers if needed
            if write_headers:
                writer.writerow(expected_headers)
            
            for item in tqdm(items, desc="Scraping Diseases"):
                disease_name = item.text.strip()
                main_link = f"https://www.mayoclinic.org{item['href']}" if item['href'].startswith("/") else item['href']

                link1, link2 = self.data_extractor(main_link)
            
            # Create a row with all expected columns
                row_data = {
                "disease": disease_name,
                "main_link": main_link,
                "Diagnosis_treatment_link": link1,
                "Doctors_departments_link": link2
            }
            
            # If appending to existing file with different headers, align data with existing headers
                if file_exists and existing_headers:
                    row = [row_data.get(header, "") for header in existing_headers]
                else:
                    row = [row_data[header] for header in expected_headers]
            
                writer.writerow(row)

        print("Scraping Completed! Data Saved")
        
if __name__ == "__main__":
    scrapper = main()
    for letter in string.ascii_uppercase:
        print(f"working on this letter {letter} ")
        scrapper.web_scraping(f"https://www.mayoclinic.org/diseases-conditions/index?letter={letter}")
        clear_output(True)
    
    df = pd.read_csv(MAYO_CSV)
    df = df.drop_duplicates(subset=["disease"])
    df.to_csv(MAYO_CSV)
# Example usage:
# web_scraping("https://www.mayoclinic.org/diseases-conditions")
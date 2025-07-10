import csv
from bs4 import BeautifulSoup
import requests
import os
import sys
from tqdm import tqdm
from IPython.display import clear_output

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from backend.config import MAYO_CSV
class next_links:
    def __init__(self):
        self.SECTION_SLUGS = [
    ("diagnosis", "Diagnosis"),
    ("treatment", "Treatment"),
    ("coping-and-support", "Coping and support"),
    ("preparing-for-your-appointment", "Preparing for your appointment"),
    ("lifestyle-and-home-remedies", "Lifestyle and home remedies")
]

    def extract_sections(self,url):
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }

        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            result_sections = {}

        # Try method 1: Old layout (div.content or article#main-content)
            main_content = soup.find('div', class_='content') or soup.find('article', id='main-content')
            if main_content:
                headings = main_content.find_all(['h2', 'h3'])
                for heading in headings:
                    heading_text = heading.get_text(strip=True)
                # Fix typo
                    if "When to see a dotor" in heading_text:
                        heading_text = "When to see a doctor"
                # If it's a section we care about
                    for _, section_name in self.SECTION_SLUGS:
                        if heading_text == section_name:
                            content = []
                            next_node = heading.find_next_sibling()
                            while next_node and next_node.name not in ['h2', 'h3']:
                                if next_node.name == 'p':
                                    content.append(next_node.get_text(strip=True))
                                elif next_node.name in ['ul', 'ol']:
                                    items = [li.get_text(strip=True) for li in next_node.find_all('li')]
                                    content.extend(items)
                                next_node = next_node.find_next_sibling()
                            if content:
                                result_sections[section_name] = '\n'.join(content)
        # Try method 2: New layout (section[aria-labelledby] + cmp-text__rich-content)
            for slug, section_name in self.SECTION_SLUGS:
                if section_name in result_sections:
                    continue  # Already found by old method
            # Find aria-labelledby section
                section = soup.find('section', {'aria-labelledby': slug})
                if section:
                    content_div = section.find('div', class_='cmp-text__rich-content')
                    if content_div:
                        paragraphs = [p.get_text(strip=True) for p in content_div.find_all('p')]
                        if paragraphs:
                            result_sections[section_name] = '\n\n'.join(paragraphs)
            return result_sections
        except Exception as e:
        # Skip on error, return empty
            return {}

    def update_csv_with_sections(self,csv_file):
    # Read existing data and headers
        rows = []
        existing_headers = []
        if os.path.exists(csv_file):
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
               reader = csv.DictReader(f)
               existing_headers = reader.fieldnames
               rows = list(reader)

        section_headers = [section for slug, section in self.SECTION_SLUGS]
        all_headers = existing_headers.copy() if existing_headers else ['disease', 'main_link']
        for header in section_headers:
            if header not in all_headers:
                all_headers.append(header)

    # --- SINGLE tqdm bar for ALL rows ---
        updated_rows = []
        with tqdm(total=len(rows), desc="Processing diseases", unit="disease") as pbar:
            for row in rows:
                if 'main_link' in row and row['Diagnosis_treatment_link']:
                    clear_output(wait=True)
                # (Optional) tqdm.write() to log messages without disrupting the bar
                    tqdm.write(f"Processing: {row.get('disease', 'Unknown')}")
                    sections = self.extract_sections(row['Diagnosis_treatment_link'])
                    for section, content in sections.items():
                       row[section] = content
                updated_rows.append(row)
                pbar.update(1)  # Always update once per disease

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_headers)
            writer.writeheader()
            writer.writerows(updated_rows)
        print(f"CSV file updated successfully with {len(updated_rows)} rows")

# Example usage
if __name__ == "__main__":
    csv_file = MAYO_CSV
    ss = next_links()
    ss.update_csv_with_sections(csv_file)
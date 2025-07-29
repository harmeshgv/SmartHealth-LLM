import re
import json
import os
import sys
from tabulate import tabulate

# Add backend path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import TEST_SCORES_JSON, README_PATH

# Load JSON data
with open(TEST_SCORES_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Generate table rows
rows = [
    [
        model.get("Version", ""),
        model.get("Model Name", ""),
        model.get("Description", ""),
        f"{model.get('Top-1 Accuracy', '')}%",
        f"{model.get('Top-3 Accuracy', '')}%",
        f"{model.get('Top-5 Accuracy', '')}%"
    ]
    for model in data.get("models", [])
]

# Table headers
headers = ["Version", "Model Name", "Description","Top-1", "Top-3", "Top-5"]

# Create markdown table
table_md = tabulate(rows, headers=headers, tablefmt="github")

# Read README content
with open(README_PATH, 'r', encoding='utf-8') as f:
    readme_content = f.read()

# Placeholders
start_placeholder = '<!--start-->'
end_placeholder = '<!--stop-->'

# Replace block
new_readme_content = re.sub(
    f"({re.escape(start_placeholder)})[\\s\\S]*({re.escape(end_placeholder)})",
    f"\\1\n{table_md}\n\\2",
    readme_content
)

# Write back to README
with open(README_PATH, 'w', encoding='utf-8') as f:
    f.write(new_readme_content)

print("✅ README successfully updated with table!")

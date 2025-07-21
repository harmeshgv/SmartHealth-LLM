import re
import json
import os
import sys

# Assuming 'backend' is in the parent directory of this script's location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming these paths are configured correctly
from backend.config import TEST_SCORES_JSON, README_PATH

# Load the JSON data from the file
# Note: JSON files are text, so they also benefit from explicit encoding
with open(TEST_SCORES_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Correctly access the nested 'project_name' value
stats_text = f"project_name = {data['data']['project_name']}"

# Read the entire content of the README file using UTF-8
with open(README_PATH, 'r', encoding='utf-8') as f:
    readme_content = f.read()

# Define the placeholders for replacement
start_placeholder = '<!--start-->'
end_placeholder = '<!--stop-->'

# Use regex to find the block between placeholders and replace its content
new_readme_content = re.sub(
    f"({re.escape(start_placeholder)})[\\s\\S]*({re.escape(end_placeholder)})",
    f"\\1\n{stats_text}\n\\2",
    readme_content
)

# Write the updated content back to the README file using UTF-8
with open(README_PATH, 'w', encoding='utf-8') as f:
    f.write(new_readme_content)

print("Readme successfully updated!")
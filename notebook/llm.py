#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# Folders to exclude
EXCLUDED_FOLDERS = {'myenv', 'scipy', '__pycache__', '.git','skin-disease-datasaet'}

def print_tree(start_path, indent=""):
    for item in sorted(os.listdir(start_path)):
        item_path = os.path.join(start_path, item)
        if item in EXCLUDED_FOLDERS:
            continue
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}")
            print_tree(item_path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

# Replace '.' with your desired root directory
print_tree('.')


# In[2]:


import pandas as pd
import spacy
import sys
import os

# In[3]:


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from backend.config import MAYO_CSV

# In[4]:


df = pd.read_csv(MAYO_CSV)

# In[5]:


df.head()

# In[6]:


symptoms1 = df["Symptoms"][0]
symptoms2 = df["Symptoms"][1]
symptoms1 , "......................................." ,symptoms2

# In[8]:


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(symptoms)

for ent in doc.ents:
    print(ent.text, ent.label_)


# In[9]:


from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "d4data/biomedical-ner-all"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

entities = ner(symptoms)
for e in entities:
    if e['entity_group'] == 'Sign_symptom':
        print(e['word'], e['entity_group'])


# In[12]:


from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
api_key = os.getenv("API_KEY")
print(api_key)


# In[14]:


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

# For Groq
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

llm = ChatOpenAI(
    model="llama3-70b-8192",  # or llama3-70b-8192 if supported
    temperature=0
)



# In[15]:


from langchain.prompts import PromptTemplate

prompt_template_name = PromptTemplate(
    input_variables=['symptoms'],
template = """
You are a medical text processor. 
Your task is to carefully read the following paragraph where a patient casually describes their health condition. 

Extract **only the medical symptoms** mentioned. 
- ❌ Do NOT include: disease names, extra text, explanations, or labels. 
- ✅ Output should be: a simple, clean, comma-separated list of symptom words/phrases, all in lowercase. 
- ✅ Remove duplicates and unnecessary words.
- ✅ Output ONLY the symptoms, nothing else.
- No extra words explaination or nothing is required , no need of " ' " this symbols too

Paragraph: {symptoms}

Symptoms (comma-separated):
"""

)


# In[16]:


chain = prompt_template_name | llm


# In[17]:


response = chain.invoke({"symptoms":symptoms2})

# In[18]:


symptom_list = [s.strip() for s in response.content.split(",")]
symptom_list

# In[19]:


import spacy

# Load the biomedical NER model
nlp = spacy.load("en_ner_bc5cdr_md")

def extract_entities(text):
    doc = nlp(text)
    symps = list({ent.text.lower() for ent in doc.ents})
    return symps
        


# In[20]:


extract_entities(symptoms1)

# In[21]:


from langchain.schema import Document

# In[22]:


disease1 = df["disease"][0]
disease2 = df["disease"][1]
disease1, disease2

# In[44]:


sympto =    extract_entities(symptoms)
sympto

# In[23]:


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# In[ ]:


embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# In[66]:


df.shape[0]

# In[69]:


for i in range(df.shape[0]):
    print(df["disease"][i])
    break

# In[71]:


df.columns

# In[ ]:


from tqdm import tqdm

disease_db = {}

for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Processing diseases"):
    disease = row["disease"]
    
    # ✅ Handle NaN / missing symptoms
    symptoms_text = row["Symptoms"]
    if not isinstance(symptoms_text, str):
        continue  # Skip rows with no symptoms

    # Extract symptoms
    symptom_list = extract_entities(symptoms_text)

    # If no symptoms extracted, skip
    if not symptom_list:
        response = chain.invoke({"symptoms": symptoms_text})
        symptom_list = [s.strip() for s in response.content.split(",")]

        

    # Generate embeddings
    symptom_embeddings = embedder.embed_documents(symptom_list)

    disease_db[disease] = {
        "symptoms": symptom_list,
        "embeddings": symptom_embeddings
    }

print(f"\n✅ Processed {len(disease_db)} diseases successfully.")


# In[ ]:


# Example structure
disease_db = {
    disease1: {
        "symptoms": symptoms1,
        "embeddings": embedder.embed_documents(extract_entities(symptoms1))
    },
    disease2: {
        "symptoms": symptoms2,
        "embeddings": embedder.embed_documents(extract_entities(symptoms2))
    }
}


# In[60]:


disease_db

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity

def match_disease(user_input, disease_db, threshold=0.7):
    query_symptoms = extract_entities(user_input)
    query_embeddings = embedder.embed_documents(query_symptoms)

    results = []

    for disease, data in disease_db.items():
        scores = cosine_similarity(query_embeddings, data["embeddings"])
        max_similarities = scores.max(axis=1)
        coverage = (max_similarities > threshold).mean()
        avg_similarity = max_similarities.mean()

        results.append((disease, coverage, avg_similarity))

    # Rank: first by coverage, then avg similarity
    results = sorted(results, key=lambda x: (-x[1], -x[2]))
    return results[:3] 


# In[88]:


df["disease"][600]

# In[89]:


match_disease(df["Symptoms"][600], disease_db)

# In[ ]:




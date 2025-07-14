
![Logo](https://github.com/harmeshgv/SmartHealth-LLM/blob/main/logo.png?raw=true)


# Smart Health LLm

**Smart Health** is a dual-module AI health assistant combining:

* 🖼️ **Skin image classification** using deep learning, and
* 💬 **Symptom-based disease prediction** using an LLM and FAISS-based vector similarity search.

It allows users to upload skin images for instant disease classification or chat naturally by describing symptoms to get possible disease predictions.

---

## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

---

## Authors

- [Harmesh G V](https://www.github.com/harmeshgv)

---

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

---

## Demo

Insert gif or link to demo

---

## Tech Stack

### Frontend

* **Streamlit** – Fast and interactive frontend for ML web apps

### Backend

* **Python** – Backend logic and ML integration
* **FastAPI** – High-performance web framework for serving APIs

### AI/ML

* **EfficientNet-B0** – Lightweight and accurate CNN for skin disease prediction
* **all-MiniLM** – Embedding model for semantic search & LLM tasks
* **RAG (Retrieval-Augmented Generation)** – Combines LLM with medical context for better answers
* **FAISS DB** – Fast similarity search for symptom/disease retrieval
* **CUDA** – GPU acceleration for faster training and inference

### DevOps / Deployment

* *(Coming Soon)* – Docker, CI/CD, and cloud deployment tools

---

## Features

- AI-Powered Diagnosis – Predicts skin diseases using EfficientNet-B0
- Symptom-based Chat – Interact with a medical assistant powered by LLM & RAG
- Fast Performance – Optimized with CUDA for GPU-based inference
- Smart Suggestions – Recommends possible causes, treatments, and follow-up questions
- Semantic Search – Finds the most relevant diseases using FAISS & MiniLM embeddings

---


## Run Locally

Clone the project

```bash
git clone https://github.com/harmeshgv/SmartHealth-LLM.git
```

Go to the project directory

```bash
cd SmartHealth-LLM
```

Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
# or
source venv/bin/activate  # For Linux/macOS
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run backend server (FastAPI)

```bash
cd backend
uvicorn main:app --reload
```

Run frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Tests

> *Not yet implemented*

---

## API Reference

> *Not yet added*

---

<pre>## Directory Structure
```
SmartHealth-LLM/
├── .gitattributes
├── .gitignore
├── LICENSE
├── README (3).md
├── README.md
├── requirements.txt
├── logo.png
├── transfaer.ipynb
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   └── upload.py
│   ├── models/
│   │   └── skin_disease_model.h5
│   ├── services/
│   │   ├── image_classifier.py
│   │   └── symptom_to_disease.py
│   └── utils/
│       ├── filtering_with_ner.py
│       ├── image_preprocessing.py
│       └── text_cleaning.py
├── data/
│   ├── labels.json
│   ├── mayo_diseases.csv
│   ├── test_symptom_cases.csv
│   └── Vector/
│       └── symptom_faiss_db/
│           ├── index.faiss
│           └── index.pkl
├── evaluation/
│   └── evaluate_rag_symptom.py
├── frontend/
│   └── index.html
├── notebook/
│   ├── llm.ipynb
│   ├── skin_disease_prediction.ipynb
│   └── web_scrapping.ipynb
├── scripts/
│   ├── model.py
│   ├── symptoms_to_vectordb.py
│   └── scrapers/
│       ├── main.py
│       ├── web_scraper2.py
│       └── web_scrapers.py
``` </pre>

---

## Documentation

[Documentation](https://linktodocumentation)

---

## Optimizations

 Model Caching – Loaded EfficientNet-B0 and LLM components once during startup to avoid reloading on each request

 CUDA Acceleration – Enabled GPU usage for faster model inference and reduced latency in skin disease prediction

 Efficient Embeddings – Used all-MiniLM to generate lightweight yet accurate embeddings for semantic search

---

## Lessons Learned

Building SmartHealth-LLM gave me hands-on experience in combining deep learning and LLM-based reasoning for real-world healthcare use cases. Some key lessons and challenges:

* **RAG & Retrieval Design:**
  Learned how to build a symptom-based semantic search system using FAISS and MiniLM, and integrate it with a language model for accurate, context-aware responses.

* **Model Integration & Performance:**
  Faced challenges integrating image-based CNN models (EfficientNet-B0) with real-time web applications. Solved it using model caching and optimizing inference with CUDA.

* **LLM Prompt Tuning:**
  Realized the importance of prompt clarity and context when working with LLMs to ensure medical suggestions were structured, useful, and safe.

* **Handling Noisy Medical Data:**
  Built text cleaning and NER filters to extract relevant keywords and symptoms from unstructured input.

* **Frontend-Backend Sync:**
  Gained experience managing interactions between a Streamlit frontend and a FastAPI backend — handling JSON inputs, form validation, and async inference calls.

* **Trade-offs in Real-Time AI Apps:**
  Learned to balance model accuracy with responsiveness and resource usage, especially in a setup where both image and text processing are involved.

---

## Roadmap

- Additional browser support

- Add more integrations

---

## Appendix

Any additional information goes here

---

## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

---

## Support

For support, email fake@fake.com or join our Slack channel.

---

## Feedback

If you have any feedback or suggestions, feel free to reach out at **harmeshgopinathan@gmail.com**

---

## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)

---

## FAQ

#### How does SmartHealth predict skin diseases?

 It uses a pre-trained EfficientNet-B0 deep learning model to classify uploaded skin images.

#### What happens when I enter my symptoms?

The system uses semantic search (FAISS + MiniLM) to find relevant diseases and then uses an LLM to generate detailed responses.

####  Is my data stored?

Currently, the app does not store any user data. All predictions happen in-memory for privacy.

#### Can I use this without a GPU?

Yes, but the training will be slower. CUDA acceleration is used only if available.

---

## Related

Here are some related projects

[CSV RAG](https://github.com/harmeshgv/RAG-Enhanced-NLP-QueryEngine.git)


# Hi, I'm Katherine! 👋


## 🚀 About Me
I'm a full stack developer...


## 🛠 Skills
Javascript, HTML, CSS...


## Other Common Github Profile Sections
👩‍💻 I'm currently working on...

🧠 I'm currently learning...

👯‍♀️ I'm looking to collaborate on...

🤔 I'm looking for help with...

💬 Ask me about...

📫 How to reach me...

😄 Pronouns...

⚡️ Fun fact...


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://harmeshgv.github.io/portfolio/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/harmeshgv)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/Harmesh950)


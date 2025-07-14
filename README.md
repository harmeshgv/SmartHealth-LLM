
![Logo](https://github.com/harmeshgv/SmartHealth-LLM/blob/main/logo.png?raw=true)


# Smart Health LLm

**Smart Health** is a dual-module AI health assistant combining:

* 🖼️ **Skin image classification** using deep learning, and
* 💬 **Symptom-based disease prediction** using an LLM and FAISS-based vector similarity search.

It allows users to upload skin images for instant disease classification or chat naturally by describing symptoms to get possible disease predictions.


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)



## Authors

- [@octokatherine](https://www.github.com/octokatherine)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Demo

Insert gif or link to demo


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


## Features

- AI-Powered Diagnosis – Predicts skin diseases using EfficientNet-B0
- Symptom-based Chat – Interact with a medical assistant powered by LLM & RAG
- Fast Performance – Optimized with CUDA for GPU-based inference
- Smart Suggestions – Recommends possible causes, treatments, and follow-up questions
- Semantic Search – Finds the most relevant diseases using FAISS & MiniLM embeddings

Perfect! Based on your format, here’s a customized and **structured README scaffold** for your **Smart Health - LLM** project, where sections you haven’t implemented yet are **left empty**, and the rest are customized to fit your Python + Streamlit + FastAPI + ML/LLM stack:

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

## Environment Variables


---

## Deployment

> *Not yet implemented*

---

## Running Tests

> *Not yet implemented*

---

## Usage/Examples

> *Not yet implemented*

---

## API Reference

> *Not yet added*

---

## Directory Structure

📄 .gitattributes
📄 .gitignore
📄 LICENSE
📄 README (3).md
📄 README.md
📁 backend
    📁 api
        📄 upload.py
    📄 config.py
    📄 main.py
    📁 models
        📄 skin_disease_model.h5
    📁 services
        📄 image_classifier.py
        📄 symptom_to_disease.py
    📁 utils
        📄 filtering_with_ner.py
        📄 image_preprocessing.py
        📄 text_cleaning.py
📁 data
    📁 Vector
        📁 symptom_faiss_db
            📄 index.faiss
            📄 index.pkl
    📄 labels.json
    📄 mayo_diseases.csv
    📄 test_symptom_cases.csv
📁 evaluation
    📄 evaluate_rag_symptom.py
📁 frontend
    📄 index.html
📄 logo.png
📁 notebook
    📄 llm.ipynb
    📄 skin_disease_prediction.ipynb
    📄 web_scrapping.ipynb
📄 requirements.txt
📁 scripts
    📄 model.py
    📁 scrapers
        📄 main.py
        📄 web_scraper2.py
        📄 web_scrapers.py
    📄 symptoms_to_vectordb.py
📄 transfaer.ipynb

---


## Documentation

[Documentation](https://linktodocumentation)


## API Reference

#### Get all items

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |

#### Get item

```http
  GET /api/items/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


## Optimizations

 Model Caching – Loaded EfficientNet-B0 and LLM components once during startup to avoid reloading on each request

 CUDA Acceleration – Enabled GPU usage for faster model inference and reduced latency in skin disease prediction

 Efficient Embeddings – Used all-MiniLM to generate lightweight yet accurate embeddings for semantic search


## Lessons Learned

What did you learn while building this project? What challenges did you face and how did you overcome them?


## Roadmap

- Additional browser support

- Add more integrations


## Appendix

Any additional information goes here


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.


## Support

For support, email fake@fake.com or join our Slack channel.


## Feedback

If you have any feedback, please reach out to us at fake@fake.com


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2


## Related

Here are some related projects

[Awesome README](https://github.com/matiassingers/awesome-readme)


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
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)


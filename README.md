<div align="center">
  <img src="./docs/assets/logo.png" alt="SmartHealth Logo" width="200"/>



  ### AI-Powered Medical Assistant with Multi-Agent Architecture for Disease Classification & Symptom Analysis

  [![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-4285F4?style=for-the-badge)](https://smart-health-llm.streamlit.app/)
  [![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/harmeshgv/SmartHealth-LLM/blob/main/LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
  [![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org/)
  [![Stars](https://img.shields.io/github/stars/harmeshgv/SmartHealth-LLM?style=for-the-badge)](https://github.com/harmeshgv/SmartHealth-LLM/stargazers)

  [🚀 Quick Start](#-quick-start) • [📚 Documentation](#-features) • [🏗️ Architecture](#-architecture) • [🤝 Contributing](#-contributing)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Performance Metrics](#-performance-metrics)
- [Screenshots](#-screenshots)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Directory Structure](#-directory-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## Overview


**SmartHealth-LLM** is an advanced **AI-powered medical assistant** that blends **computer vision**, **natural language processing**, and a **multi-agent reasoning framework** to deliver intelligent, reliable health assessments. The system can interpret both textual symptoms and medical images, offering AI-backed insights to assist in early disease detection and medical understanding. Designed with modular intelligence, each agent performs specialized roles such as interpreting user input, analyzing skin images, retrieving verified medical information, and deciding the next best action.

Key Highlights:

1. **Multi-Agent Symptom Analysis** – A collaborative agent ecosystem that uses deep learning and reasoning to interpret symptoms, correlate them with diseases, and guide diagnostic flows.
2. **Semantic Disease Matching** – FAISS-powered vector search enables high-precision similarity matching between patient symptoms and medical knowledge bases.
3. **AI-Enhanced Image Diagnosis** – A computer vision pipeline using **DenseNet** for accurate skin disease classification and visual health assessments.



### Why SmartHealth-LLM?

- **Multi-Agent System**: Specialized agents (Symptom, Disease Info, Decider, Formatter) for intelligent orchestration
- **Real-Time**: Optimized inference pipeline with CUDA support for instant predictions
- **Privacy-First**: In-memory processing with no persistent data storage
- **Modern Stack**: React + TypeScript frontend, FastAPI backend, Streamlit interface
- **Research-Grade**: Leverages LangChain, FAISS vector databases, biomedical NER, and state-of-the-art embeddings

---

## Key Features

### Dual-Mode Health Assessment

| Feature | Description | Technology |
|---------|-------------|------------|
| **Skin Disease Classification** | Upload images for instant AI-powered diagnosis through agents| EfficientNet-B0 CNN |
| **Symptom Analysis** | Chat with AI to analyze symptoms and get disease predictions | LLM + RAG Pipeline |
| **Semantic Search** | Find diseases matching your symptoms using vector similarity | FAISS + MiniLM |
| **Multi-Agent Architecture** | Separate agents for classification, chat, and verification | LangChain + LangGraph |

### Performance Highlights

- **99.86%** Top-5 Disease Matching Accuracy
- **GPU-Accelerated** inference with CUDA support
- **Real-Time** predictions and chat responses
- **200K+ Context** window for comprehensive medical knowledge

---

## Architecture

### System Design Overview

[![](https://mermaid.ink/img/pako:eNptkk2TojAQhv9KKmfHFQURDluloo4z4_fMZYOHFLSYWkjYELbWVf_7xiAuUzWc6LzP293p9BlHIgbs40TS_IjeRyFH-huSLdBIoakUXAGP0Te0UxJoljKFPuZ79PT0HY3OW_hVQqHQVpQK5LXyjm7iZXfKciUytClBni5oTIYJcIVWMjpqh6RKyH2Tn2c0AfSRp4LGFxSQKh6ntCjYgUVUMcHRmuWQMg7aWXnHppEJqauZIvumNiUBK4AWgOb8IL4AZiSAiMUgv9CeyVTIjCrVUCt9YvQ5GTGRQaz7S9FyskXvQqT7JvHyKL-gSl9dNpGpQV4_d7gFJRn8BvkJeiMzIZIU0A6oHmGdpUJeDLIg0-F8t0P1LILRPcOrkZd3uS5m5AoIDLAi4-USLfQ6_M88M8r6fBtRcXuBN5Gw6P7Q62pGVfBsgs1jYLG-SJELXsC9i1UF1Jk3JhyGHLf07rEY-0qW0MIZaP8txOcbGGJ1hAxC7OvfmMqfIQ75VXtyyn8IkdU2KcrkiP0DTQsdlXlMFQSM6q3OHqdSLzLIsSi5wn6365gk2D_jP9i3PKvtOZ3OwOq7tuvYvRY-6dO-1x54juf2Bl3Pdbr2tYX_mqqdtms7Pdt2LK9vOZbb613_Adq1_9c?type=png)](https://mermaid.live/edit#pako:eNptkk2TojAQhv9KKmfHFQURDluloo4z4_fMZYOHFLSYWkjYELbWVf_7xiAuUzWc6LzP293p9BlHIgbs40TS_IjeRyFH-huSLdBIoakUXAGP0Te0UxJoljKFPuZ79PT0HY3OW_hVQqHQVpQK5LXyjm7iZXfKciUytClBni5oTIYJcIVWMjpqh6RKyH2Tn2c0AfSRp4LGFxSQKh6ntCjYgUVUMcHRmuWQMg7aWXnHppEJqauZIvumNiUBK4AWgOb8IL4AZiSAiMUgv9CeyVTIjCrVUCt9YvQ5GTGRQaz7S9FyskXvQqT7JvHyKL-gSl9dNpGpQV4_d7gFJRn8BvkJeiMzIZIU0A6oHmGdpUJeDLIg0-F8t0P1LILRPcOrkZd3uS5m5AoIDLAi4-USLfQ6_M88M8r6fBtRcXuBN5Gw6P7Q62pGVfBsgs1jYLG-SJELXsC9i1UF1Jk3JhyGHLf07rEY-0qW0MIZaP8txOcbGGJ1hAxC7OvfmMqfIQ75VXtyyn8IkdU2KcrkiP0DTQsdlXlMFQSM6q3OHqdSLzLIsSi5wn6365gk2D_jP9i3PKvtOZ3OwOq7tuvYvRY-6dO-1x54juf2Bl3Pdbr2tYX_mqqdtms7Pdt2LK9vOZbb613_Adq1_9c)

### Multi-Agent Architecture

The platform implements a sophisticated **agent orchestration system** with specialized AI agents:

#### **Agent Orchestrator**
Central coordinator that manages agent communication, workflow execution, and decision routing.

#### Core Agents:

**Symptom Agent**
Processes natural language symptom descriptions to extract key medical entities using biomedical NER. It queries the FAISS symptom vector database for semantic matching and collaborates with the Disease Matcher Tool for multi-stage disease retrieval.

**Disease Info Agent**
Retrieves comprehensive disease information by querying the FAISS disease vector database and integrating Google Search for the latest medical updates. It provides detailed disease profiles along with relevant metadata.

**Decider Agent**
Acts as the central routing intelligence, deciding which agents to invoke based on the query type. It handles edge cases, uncertainty, and optimizes agent workflows for efficient processing.

**Formatter Agent**
Structures responses for clarity and readability, formatting medical information with proper citations. Ensures consistent output across all agents and generates user-friendly explanations.

**Image Diagnosis Agent**
Analyzes medical images, particularly for skin diseases, using deep learning models. It complements text-based diagnosis by providing visual detection and classification of conditions.


#### Specialized Tools:

- **Biomedical NER Tool** - Extracts medical entities (symptoms, diseases, body parts)
- **Disease Matcher Tool** - Semantic similarity search using FAISS
- **Disease Info Retriever** - Comprehensive disease knowledge retrieval
- **Google Search Tool** - Real-time medical information augmentation
- **Skin Disease Prediction Tool** - Skin Disease Image Classification Using DenseNet121 Model

### Technology Stack Layers
| Layer                     | Technologies / Components                        |
|----------------------------|-------------------------------------------------|
| Frontend Layer             | React 18+, TypeScript, Streamlit, Chat, Image Upload |
| Orchestration Layer        | Agent Orchestrator, Multi-Agent Coordination, Workflow Management |
| Agent Layer                | Symptom Agent, Disease Info Agent, Decider Agent, Formatter Agent, LangChain Integration |
| Tools & Services Layer     | Biomedical NER, Disease Matcher, Info Retriever, Google Search |
| Data & ML Layer            | FAISS Vector DBs, MiniLM Embeddings, Medical Knowledge Base, DenseNet121 Model |



---

## Performance Metrics

### Disease Matching Evolution

coming soon

---

## Screenshots

### Skin Disease Classification

*AI-powered skin disease detection with confidence scores and treatment recommendations*

### Multi-Agent Chatbot Interface

*Intelligent symptom analysis powered by multi-agent orchestration*

### System Architecture

*Multi-agent architecture with specialized AI agents and FAISS vector databases*

---

## 🛠️ Tech Stack

### Frontend
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)

### Backend & Orchestration
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜_LangChain-000000?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-034000?style=flat-square)

### AI/ML & Vector Search
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0066FF?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=flat-square)
![DenseNet121](https://img.shields.io/badge/DenseNet121-119988)


### DevOps & Testing
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (optional, for faster inference)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/harmeshgv/SmartHealth-LLM.git
cd SmartHealth-LLM
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Add your API keys
GROQ_API_KEY=your_key_here
GROQ_API_BASE=base_link
GROQ_API_MODEL=selected_model
# or
GRAVIX_API_KEY=your_key_here
GRAVIX_API_BASE=base_link
GRAVIXs_API_MODEL=selected_model
```

---

## Usage

### Running the Application

#### Option 1: Streamlit Interface (Recommended for Demo)
```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

#### Option 2: React Frontend + Python Backend

**Terminal 1 - Start Backend**
```bash
# Run main application
uvicorn backend.main:app --reload

# Backend runs on http://localhost:8000
```

**Terminal 2 - Start React Frontend**
```bash
cd frontend
npm install
npm start

# Frontend runs on http://localhost:3000
```

### Using Docker

```bash
# Build and run with Docker
docker build -t smarthealth-llm .
docker run -p 8501:8501 smarthealth-llm

# Access the app at http://localhost:8501
```

### Building FAISS Vector Databases

If you need to rebuild the vector databases:

```bash
# Build symptom vector database
python scripts/build_faiss_db.py

# Build disease information database
python scripts/build_disease_db.py
```

---

## Directory Structure

```
SmartHealth-LLM/
│
├── 📄 README.md                    # Project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 pyproject.toml               # Project configuration
├── 📄 Dockerfile                   # Docker configuration
├── 📄 .env                         # Environment variables
├── 📄 main.py                      # Main application entry
├── 📄 streamlit_app.py             # Streamlit web interface
│
├── ⚙️ backend/                     # Backend system
│   ├── __init__.py
│   ├── README.md                   # Backend documentation
│   ├── config.py                   # Configuration settings
│   ├── agent_orchestrator.py      # Multi-agent coordinator
│   │
│   ├── 🤖 agents/                  # Specialized AI agents
│   │   ├── __init__.py
│   │   ├── symptom_agent.py       # Symptom analysis agent
│   │   ├── disease_info_agent.py  # Disease information agent
│   │   ├── decider_agent.py       # Decision routing agent
│   │   └── formatter_agent.py     # Response formatting agent
│   │
│   ├── 🛠️ tools/                   # Agent tools
│   │   ├── __init__.py
│   │   ├── biomedical_ner_tool.py # Medical entity extraction
│   │   ├── disease_matcher_tool.py # FAISS-based disease matching
│   │   ├── disease_info_retriever_tool.py # Disease info retrieval
│   │   └── google_search.py       # Web search integration
│   │
│   ├── 📊 data/                    # Datasets and vector stores
│   │   ├── __init__.py
│   │   ├── updated_df.csv         # Medical knowledge base
│   │   ├── labels.json            # Disease labels
│   │   ├── test.json              # Test data
│   │   └── Vector/                # FAISS vector databases
│   │       ├── symptom_faiss_db/  # Symptom embeddings
│   │       │   ├── index.faiss
│   │       │   └── index.pkl
│   │       └── disease_faiss_db/  # Disease embeddings
│   │           ├── index.faiss
│   │           └── index.pkl
│   │
│   └── 🔧 utils/                   # Utility modules
│       ├── __init__.py
│       ├── llm.py                 # LLM configurations
│       └── embeddings.py          # Embedding models
│
├── 🎨 frontend/                    # React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── public/                    # Static assets
│   │   ├── index.html
│   │   ├── manifest.json
│   │   ├── favicon.ico
│   │   ├── logo192.png
│   │   └── logo512.png
│   └── src/                       # React components
│       ├── index.tsx
│       ├── App.tsx
│       ├── App.css
│       ├── ChatPage.tsx
│       ├── ChatPage.css
│       ├── api.ts                 # API integration
│       └── index.css
│
├── 📓 notebook/                    # Jupyter notebooks
│   ├── __init__.py
│   ├── skin_disease_prediction.ipynb # CNN training
│   ├── web_scrapping.ipynb        # Data collection
│   └── catboost_info/             # Model training logs
│
├── 📜 scripts/                     # Utility scripts
│   ├── model.py                   # Model utilities
│   ├── build_faiss_db.py          # Build symptom vector DB
│   ├── build_disease_db.py        # Build disease vector DB
│   ├── update_readme.py           # Documentation generator
│   └── scrapers/                  # Web scraping tools
│       ├── main.py
│       ├── web_scrapers.py
│       └── web_scraper2.py
│
├── 🧪 tests/                       # Unit tests
│   └── test_llm.py                # LLM integration tests
│
├── 📖 docs/                        # Documentation
│   └── assets/                    # Images and media
│       ├── logo.png
│       ├── skin.png
│       ├── chatbot.png
│       └── image.png
│
└── 🔄 .github/                     # GitHub workflows
    └── workflows/
        └── python-test.yml        # CI/CD pipeline
```

---

## Testing

### Run Unit Tests
```bash
pytest tests/
```

### Run Integration Tests
```bash
pytest tests/integration/
```

### Test Coverage
```bash
pytest --cov=backend tests/
```

---

## 🔧 Configuration

### Environment Variables

coming soon

---

## Deployment

coming soon

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
-  Report bugs and issues
-  Suggest new features
-  Improve documentation
-  Submit pull requests

### Contribution Guidelines

1. **Fork the repository**
```bash
git fork https://github.com/harmeshgv/SmartHealth-LLM.git
```

2. **Create a feature branch**
```bash
git checkout -b feature/AmazingFeature
```

3. **Commit your changes**
```bash
git commit -m 'Add some AmazingFeature'
```

4. **Push to the branch**
```bash
git push origin feature/AmazingFeature
```

5. **Open a Pull Request**

### Code Style
- Follow PEP 8 for Python code
- Use type hints where applicable
- Write docstrings for functions
- Add unit tests for new features

---

## Learning Resources

- **Notebooks**: Check the \`notebook/\` directory for detailed tutorials
- **API Docs**: See \`docs/API.md\` for endpoint documentation
- **Blog Posts**: [Coming Soon]
- **Video Tutorials**: [Coming Soon]

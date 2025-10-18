export const readmeContent = `<div align="center">
  <img src="./docs/assets/logo.png" alt="SmartHealth Logo" width="200"/>

  # 🏥 SmartHealth-LLM

  ### AI-Powered Medical Assistant with Multi-Agent Architecture for Disease Classification & Symptom Analysis

  [![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-4285F4?style=for-the-badge)](https://smart-health-llm.streamlit.app/)
  [![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/harmeshgv/SmartHealth-LLM/blob/main/LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
  [![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org/)
  [![Stars](https://img.shields.io/github/stars/harmeshgv/SmartHealth-LLM?style=for-the-badge)](https://github.com/harmeshgv/SmartHealth-LLM/stargazers)

  [🚀 Quick Start](#-quick-start) • [📚 Documentation](#-features) • [🏗️ Architecture](#-architecture) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

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

## 🌟 Overview

**SmartHealth-LLM** is a cutting-edge AI-powered medical assistant that combines **computer vision** and **natural language processing** with a sophisticated **multi-agent architecture** to provide intelligent health assessments. The platform features:

1. **Multi-Agent Symptom Analysis** - Intelligent agent orchestration for accurate symptom-based diagnosis and skin disease detection using deep learning
2. **Semantic Disease Matching** - FAISS-powered vector search for precise disease identification
3. **Modern UI** - React frontend with Streamlit backend for seamless user experience

### Why SmartHealth-LLM?

- **Multi-Agent System**: Specialized agents (Symptom, Disease Info, Decider, Formatter) for intelligent orchestration
- **Real-Time**: Optimized inference pipeline with CUDA support for instant predictions
- **Privacy-First**: In-memory processing with no persistent data storage
- **Modern Stack**: React + TypeScript frontend, FastAPI backend, Streamlit interface
- **Research-Grade**: Leverages LangChain, FAISS vector databases, biomedical NER, and state-of-the-art embeddings

---

## 🎯 Key Features

### 🔬 Dual-Mode Health Assessment

| Feature | Description | Technology |
|---------|-------------|------------|
| **Skin Disease Classification** | Upload images for instant AI-powered diagnosis through agents| EfficientNet-B0 CNN |
| **Symptom Analysis** | Chat with AI to analyze symptoms and get disease predictions | LLM + RAG Pipeline |
| **Semantic Search** | Find diseases matching your symptoms using vector similarity | FAISS + MiniLM |
| **Multi-Agent Architecture** | Separate agents for classification, chat, and verification | LangChain + LangGraph |

### ⚡ Performance Highlights

- **99.86%** Top-5 Disease Matching Accuracy
- **GPU-Accelerated** inference with CUDA support
- **Real-Time** predictions and chat responses
- **200K+ Context** window for comprehensive medical knowledge

### 🎨 User Experience

- Modern, responsive UI with gradient designs
- Real-time chat with streaming responses
- Confidence scores and visual feedback
- Mobile-friendly interface

---

## 🏗️ Architecture

### System Design Overview

[![](https://mermaid.ink/img/pako:eNptkk2TojAQhv9KKmfHFQURDluloo4z4_fMZYOHFLSYWkjYELbWVf_7xiAuUzWc6LzP293p9BlHIgbs40TS_IjeRyFH-huSLdBIoakUXAGP0Te0UxJoljKFPuZ79PT0HY3OW_hVQqHQVpQK5LXyjm7iZXfKciUytClBni5oTIYJcIVWMjpqh6RKyH2Tn2c0AfSRp4LGFxSQKh6ntCjYgUVUMcHRmuWQMg7aWXnHppEJqauZIvumNiUBK4AWgOb8IL4AZiSAiMUgv9CeyVTIjCrVUCt9YvQ5GTGRQaz7S9FyskXvQqT7JvHyKL-gSl9dNpGpQV4_d7gFJRn8BvkJeiMzIZIU0A6oHmGdpUJeDLIg0-F8t0P1LILRPcOrkZd3uS5m5AoIDLAi4-USLfQ6_M88M8r6fBtRcXuBN5Gw6P7Q62pGVfBsgs1jYLG-SJELXsC9i1UF1Jk3JhyGHLf07rEY-0qW0MIZaP8txOcbGGJ1hAxC7OvfmMqfIQ75VXtyyn8IkdU2KcrkiP0DTQsdlXlMFQSM6q3OHqdSLzLIsSi5wn6365gk2D_jP9i3PKvtOZ3OwOq7tuvYvRY-6dO-1x54juf2Bl3Pdbr2tYX_mqqdtms7Pdt2LK9vOZbb613_Adq1_9c?type=png)](https://mermaid.live/edit#pako:eNptkk2TojAQhv9KKmfHFQURDluloo4z4_fMZYOHFLSYWkjYELbWVf_7xiAuUzWc6LzP293p9BlHIgbs40TS_IjeRyFH-huSLdBIoakUXAGP0Te0UxJoljKFPuZ79PT0HY3OW_hVQqHQVpQK5LXyjm7iZXfKciUytClBni5oTIYJcIVWMjpqh6RKyH2Tn2c0AfSRp4LGFxSQKh6ntCjYgUVUMcHRmuWQMg7aWXnHppEJqauZIvumNiUBK4AWgOb8IL4AZiSAiMUgv9CeyVTIjCrVUCt9YvQ5GTGRQaz7S9FyskXvQqT7JvHyKL-gSl9dNpGpQV4_d7gFJRn8BvkJeiMzIZIU0A6oHmGdpUJeDLIg0-F8t0P1LILRPcOrkZd3uS5m5AoIDLAi4-USLfQ6_M88M8r6fBtRcXuBN5Gw6P7Q62pGVfBsgs1jYLG-SJELXsC9i1UF1Jk3JhyGHLf07rEY-0qW0MIZaP8txOcbGGJ1hAxC7OvfmMqfIQ75VXtyyn8IkdU2KcrkiP0DTQsdlXlMFQSM6q3OHqdSLzLIsSi5wn6365gk2D_jP9i3PKvtOZ3OwOq7tuvYvRY-6dO-1x54juf2Bl3Pdbr2tYX_mqqdtms7Pdt2LK9vOZbb613_Adq1_9c)

### Multi-Agent Architecture

The platform implements a sophisticated **agent orchestration system** with specialized AI agents:

#### 🎯 **Agent Orchestrator**
Central coordinator that manages agent communication, workflow execution, and decision routing.

#### Core Agents:

1. **💡 Symptom Agent** (symptom_agent.py)
   - Processes natural language symptom descriptions
   - Extracts medical entities using biomedical NER
   - Queries FAISS symptom vector database for semantic matching
   - Collaborates with Disease Matcher Tool for multi-stage retrieval

2. **📚 Disease Info Agent** (disease_info_agent.py)
   - Retrieves comprehensive disease information
   - Queries FAISS disease vector database
   - Integrates Google Search for latest medical information
   - Provides detailed disease profiles and metadata

3. **🧠 Decider Agent** (decider_agent.py)
   - Makes intelligent routing decisions
   - Determines which agents to invoke based on query type
   - Handles edge cases and uncertainty
   - Optimizes agent workflow for efficiency

4. **✨ Formatter Agent** (formatter_agent.py)
   - Structures responses for readability
   - Formats medical information with proper citations
   - Ensures consistent output across all agents
   - Generates user-friendly explanations

#### 🛠️ Specialized Tools:

- **Biomedical NER Tool** - Extracts medical entities (symptoms, diseases, body parts)
- **Disease Matcher Tool** - Semantic similarity search using FAISS
- **Disease Info Retriever** - Comprehensive disease knowledge retrieval
- **Google Search Tool** - Real-time medical information augmentation

### Technology Stack Layers
| Layer                     | Technologies / Components                        |
|----------------------------|-------------------------------------------------|
| Frontend Layer             | React 18+, TypeScript, Streamlit, Chat, Image Upload |
| Orchestration Layer        | Agent Orchestrator, Multi-Agent Coordination, Workflow Management |
| Agent Layer                | Symptom Agent, Disease Info Agent, Decider Agent, Formatter Agent, LangChain Integration |
| Tools & Services Layer     | Biomedical NER, Disease Matcher, Info Retriever, Google Search |
| Data & ML Layer            | FAISS Vector DBs, CNN Models, MiniLM Embeddings, Medical Knowledge Base |



---

## 📊 Performance Metrics

### Disease Matching Evolution

| Version | Approach | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy |
|---------|----------|----------------|----------------|----------------|
| **v1** | Basic FAISS + NER | 8.00% | 15.00% | 18.00% |
| **v2** | Enhanced Preprocessing | 16.45% | 26.35% | 30.71% |
| **v3** | Symptom Embeddings | 92.86% | 94.90% | 95.71% |
| **v4** | LlamaIndex + MiniLM | **99.71%** | **99.85%** | **99.86%** ✨ |

### Model Specifications

| Component | Model | Parameters | Speed | Use Case |
|-----------|-------|------------|-------|----------|
| **Image Classifier** | EfficientNet-B0 | 5.3M | ~50ms | Skin disease detection |
| **Embeddings** | all-MiniLM-L6-v2 | 22.7M | ~20ms | Semantic search |
| **LLM** | GPT/Gemini | - | Variable | Conversational responses |
| **Vector DB** | FAISS | - | <10ms | Similarity search |

---

## 📸 Screenshots

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

### AI/ML & Vector Search
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0066FF?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=flat-square)

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
OPENAI_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here
```

---

## 💻 Usage

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
python main.py

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

## 📁 Directory Structure

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

## 🧪 Testing

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

```env
# API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Model Settings
MODEL_PATH=backend/models/efficientnet_b0.h5
VECTOR_DB_PATH=data/vector/

# Server Config
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

---

## 🚀 Deployment

### Deploy to Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

### Deploy with Docker
```bash
docker build -t smarthealth-llm .
docker run -p 8501:8501 smarthealth-llm
```

### Deploy to Cloud Platforms
- **AWS**: EC2, ECS, or Lambda
- **GCP**: Cloud Run or App Engine
- **Azure**: App Service or Container Instances

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

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

## 🎓 Learning Resources

- **Notebooks**: Check the \`notebook/\` directory for detailed tutorials
- **API Docs**: See \`docs/API.md\` for endpoint documentation
- **Blog Posts**: [Coming Soon]
- **Video Tutorials**: [Coming Soon]

---

## 🔒 Privacy & Security

- **No Data Storage**: All predictions happen in-memory
- **HIPAA Considerations**: Not intended for clinical use
- **API Security**: Rate limiting and authentication implemented
- **Data Encryption**: In transit via HTTPS

---

## 🐛 Known Issues & Limitations

- Model works best with high-quality skin images
- Symptom analysis requires clear, detailed descriptions
- GPU recommended for optimal performance
- Not a replacement for professional medical advice

---

## 📈 Roadmap

### Phase 1 (Current) ✅
- [x] Skin disease classification
- [x] Symptom-based chat
- [x] FAISS vector search
- [x] Streamlit interface

### Phase 2 (In Progress) 🚧
- [ ] Multi-language support
- [ ] Voice input capability
- [ ] Treatment recommendations
- [ ] Drug interaction checker

### Phase 3 (Planned) 📋
- [ ] Mobile app (React Native)
- [ ] Telemedicine integration
- [ ] Electronic health records (EHR) integration
- [ ] Real-time collaboration features

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Harmesh G V

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📞 Contact & Support

<div align="center">

### 👨‍💻 Harmesh G V

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://harmeshgv.github.io/portfolio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harmeshgv)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/Harmesh950)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:harmeshgopinathan@gmail.com)

**B.Tech Computer Science & Engineering Student**
*Passionate about AI-Powered Healthcare Solutions*

</div>

---

## 🙏 Acknowledgments

- **Mayo Clinic** - Disease database
- **HuggingFace** - Pre-trained models
- **Streamlit** - Amazing web framework
- **FastAPI** - High-performance backend
- **LangChain** - LLM orchestration
- **Community Contributors** - Thank you! 🎉

---

## 📚 Citations

If you use this project in your research, please cite:

```bibte
@software{smarthealth_llm,
  author = {Harmesh G V},
  title = {SmartHealth-LLM: AI-Powered Medical Assistant},
  year = {2025},
  url = {https://github.com/harmeshgv/SmartHealth-LLM}
}
```

---
## 📊 Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/harmeshgv/SmartHealth-LLM)
![GitHub language count](https://img.shields.io/github/languages/count/harmeshgv/SmartHealth-LLM)
![GitHub top language](https://img.shields.io/github/languages/top/harmeshgv/SmartHealth-LLM)
![GitHub last commit](https://img.shields.io/github/last-commit/harmeshgv/SmartHealth-LLM)
![GitHub issues](https://img.shields.io/github/issues/harmeshgv/SmartHealth-LLM)
![GitHub pull requests](https://img.shields.io/github/issues-pr/harmeshgv/SmartHealth-LLM)

---

<div align="center">

### ⭐ If you find this project helpful, please consider giving it a star!

**Made with ❤️ by [Harmesh G V](https://github.com/harmeshgv)**

[🔝 Back to Top](#-smarthealth-llm)

</div>`;

# 🧠 Smart Health: AI-Powered Skin Disease Classifier & Symptom Checker

## 🚀 Overview

**Smart Health** is a dual-module AI health assistant combining:

* 🖼️ **Skin image classification** using deep learning, and
* 💬 **Symptom-based disease prediction** using an LLM and FAISS-based vector similarity search.

It allows users to upload skin images for instant disease classification or chat naturally by describing symptoms to get possible disease predictions.

---

## 🌟 Features

### 🖼️ Skin Disease Image Classification

* Upload an image of a skin condition to get:

  * Predicted disease class (e.g., eczema, psoriasis)
  * Model confidence score
* Models used:

  * `MobileNetV2`, `EfficientNet-B0` trained on 8 skin disease categories

### 💬 Symptom-Based LLM Chatbot

* Enter symptoms (e.g., *"rash and itching behind knees"*) in natural language
* Backend cleans and filters symptoms using:

  * Custom NER (`biomedical-ner-all`)
  * Text preprocessing pipeline
* Matches symptoms against diseases using **FAISS + Sentence Transformers**
* Returns:

  * Top 3 closest matching diseases
  * Key symptoms & confidence scores

### 🏥 Disease Info & Q\&A

* Backend integrates with LLM to generate:

  * Explanation of the disease
  * Causes, complications, prevention
  * When to see a doctor

---

## 📈 Model Performance

### MobileNetV2

* ✅ Validation Accuracy: 92.7%
* ✅ Test Accuracy: \~90.99%

### EfficientNet-B0

* ✅ Test Accuracy: 93.16%
* ✅ Final Loss: 0.0872

---

## 🧰 Tech Stack

| Layer                   | Tools                                  |
| ----------------------- | -------------------------------------- |
| **Frontend**            | HTML form + Streamlit (for testing)    |
| **Backend**             | FastAPI, TensorFlow/Keras, PyTorch     |
| **Embeddings**          | `all-MiniLM-L6-v2` via HuggingFace     |
| **Vector Store**        | FAISS (Symptom-to-Disease Matching)    |
| **NER & Text Cleaning** | Biomedical NER + Custom pipeline       |
| **Web Scraping**        | BeautifulSoup (Mayo Clinic extraction) |

---

## 🛠️ Setup & Installation

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/SmartHealth-LLM.git
cd SmartHealth-LLM
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models

* Place the skin classification `.h5` or `.pth` model in `backend/models/`
* Ensure the FAISS index exists at:

  ```
  backend/Vector/symptom_faiss_db/index.faiss
  ```

> 🔁 If not available, run a script to generate it from your CSV or dataset.

### 4. Run the FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

### 5. Open in Browser

Go to [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## ✨ Usage

### 🖼️ Image Upload

* Upload a skin image
* Receive predicted disease and confidence

### 💬 Symptom Checker Chat

* Type symptoms in the text box (e.g., *"painful rash on arms"*)
* System processes symptoms, matches with FAISS DB, and returns:

  * Top 3 matching diseases
  * Key symptoms
  * Similarity confidence

---

## 📁 Project Structure

```
SmartHealth-LLM/
├── backend/
│   ├── api/
│   │   └── upload.py
│   ├── models/
│   │   └── skin_disease_model.h5
│   ├── services/
│   │   ├── image_classifier.py
│   │   └── symptom_to_disease.py
│   ├── utils/
│   │   ├── text_cleaning.py
│   │   └── filtering_with_ner.py
│   ├── Vector/
│   │   └── symptom_faiss_db/
│   │       └── index.faiss
│   └── main.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

---

## 📚 Data Sources

* [Mayo Clinic](https://www.mayoclinic.org/)
* Public datasets (ISIC, DermNet)
* Medical papers for symptom-disease mappings

---

## ⚠️ Disclaimer

> This tool is **not a substitute for professional medical advice**. It is intended for educational and research use only. Always consult a licensed physician for diagnosis and treatment.

---

## 🤝 Contributing

Pull requests, issues, and feature requests are welcome!
Start by forking and submitting a PR or open an issue.

---

## 📜 License

MIT License – see [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Harmesh G V**
✉️ [harmeshgopinathan@gmail.com](mailto:harmeshgopinathan@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/harmeshgv)
🔗 [GitHub](https://github.com/harmeshgv)

---
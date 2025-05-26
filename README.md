# Smart Health: Skin Diseases Classification & Symptom-based Disease Prediction LLM Chatbot

## Overview

**Smart Health** is an integrated AI-driven platform designed to assist users in early detection and understanding of skin diseases. The project leverages deep learning for skin disease image classification and a Large Language Model (LLM) chatbot for symptom-based disease prediction and Q&A. This dual approach empowers users with both image-based diagnostics and conversational health support.

---

## Features

- 🖼️ **Skin Disease Image Classification**

  - Upload skin images to get instant predictions of possible skin conditions.
  - Uses state-of-the-art deep learning (e.g., MobileNetV2, EfficientNet) trained on 8+ skin disease classes.
  - Provides confidence scores for predictions.

- 💬 **Symptom-based Disease Prediction Chatbot**

  - Enter symptoms in natural language to receive possible disease matches.
  - Powered by a Large Language Model (LLM) fine-tuned for medical Q&A.
  - Provides additional information: causes, prevention, risk factors, complications, and when to see a doctor.

- 🏥 **Conversational Health Guidance**

  - Offers explanations of diseases, symptoms, and next steps.
  - Extracts trusted information from sources like Mayo Clinic using web scraping.

- 📊 **Analytics & Explainability**
  - Displays prediction confidence and highlights most probable conditions.
  - Visualizes key symptom-disease relationships.

---

## Model Performance

### Skin Disease Image Classifier Results

- **MobileNetV2**

  - **Best Validation Accuracy:** 92.70%
  - **Final Epochs:**
    - Epoch 10/10:
      - Training Accuracy: 97.69%
      - Validation Accuracy: 90.99%
      - Training Loss: 0.0764
      - Validation Loss: 0.2751

- **EfficientNet-B0**
  - **Test Accuracy:** 93.16%
  - **Final Loss:** 0.0872

_Both models were trained and validated on 8 skin disease classes using transfer learning and data augmentation._

---

## Tech Stack

- **Backend:** Python, TensorFlow/Keras, PyTorch, FastAPI/Flask (for serving models), BeautifulSoup (for scraping)
- **Frontend:** Streamlit (for web UI)
- **LLM:** HuggingFace Transformers, or similar
- **Data:** Custom-labeled skin disease images, Mayo Clinic, and other reputable medical sources

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-health-skin-diseases.git
cd smart-health-skin-diseases
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download/Train Models

- **Image Classifier:** Download the pre-trained model (`skin_disease_model_8_classes.h5` or `skin_disease_efficientnet8.pth`) and place it in the project root.
- **LLM Model:** Configure your API key or download the local LLM checkpoint as required.

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## Usage

### Skin Disease Detection

1. Open the web app.
2. Upload a clear image of the skin condition.
3. Get instant prediction and confidence score.

### Symptom Chatbot

1. Enter your symptoms or health query in the chatbot.
2. Get disease suggestions, explanations, and next steps.

---

<!--
## File Structure

```
.
├── app.py                              # Streamlit frontend
├── skin_disease_predictor.py           # Image preprocessing & prediction backend
├── chatbot_backend.py                  # LLM-based QA and symptom prediction
├── extract_mayo_sections_flexible.py   # Mayo Clinic web scraping utility
├── requirements.txt
├── README.md
└── models/
    ├── skin_disease_model_8_classes.h5
    └── (other model files)
```
-->

---

## Data Sources

- [Mayo Clinic](https://www.mayoclinic.org/)
- Public skin disease datasets (e.g., ISIC, DermNet)
- Symptom-disease mappings curated from medical literature

---

## Disclaimer

This tool is for educational and preliminary informational purposes only. It **does not provide medical advice**. Always consult a healthcare professional for diagnosis and treatment.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

[MIT License](LICENSE)

---

## Contact

For questions or collaborations, contact [harmeshgopinathan@gmail.com](harmeshgopinathan@gmail.com)

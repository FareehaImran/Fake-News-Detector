# 📰 Fake News & Misinformation Detector

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview
An interactive web application that detects fake news using four 
traditional machine learning classifiers with TF-IDF feature extraction. 
The system includes a full comparative analysis dashboard and LIME 
explainability to show which words influenced each prediction.

---

## ✨ Features
- 🔍 Real-time fake news detection with adjustable decision threshold
- 📊 Side-by-side comparison of 4 ML classifiers
- 🧠 LIME explainability — see which words drove the prediction
- 📈 ROC curves, metrics charts, and training time visualization
- 🎯 Live LIME explanation for any user-provided text

---

## 🤖 Classifiers Compared
| Classifier | Accuracy | F1-Score | Train Time |
|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 0.1s |
| Support Vector Machine | 0.9975 | 0.9975 | 0.2s |
| Passive Aggressive | 1.0000 | 1.0000 | 0.1s |
| Random Forest | 1.0000 | 1.0000 | 2.7s |

---

## 📁 Project Structure
Fake-News-Detector/
│
├── src/
│   ├── streamlit_app.py      # Main Streamlit web application
│   ├── train_model.py        # Training pipeline for all classifiers
│   ├── detect_fake_news.py   # Core detection logic
│   ├── text_clean.py         # Text preprocessing utilities
│   └── utils.py              # Helper functions
│
├── outputs/                  # Generated after training
│   ├── pipeline.joblib
│   ├── model.joblib
│   ├── vectorizer.joblib
│   ├── metrics.json
│   └── charts/
│
├── data/                     # Add dataset here (not included)
│   ├── True.csv
│   └── Fake.csv
│
├── requirements.txt
├── run_app.py
└── README.md

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download the ISOT Fake News Dataset from Kaggle:  
🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  
Place `True.csv` and `Fake.csv` inside the `data/` folder.

### 5. Train the model
```bash
python src/train_model.py --real data/True.csv --fake data/Fake.csv
```

### 6. Run the app
```bash
streamlit run src/streamlit_app.py
```
Open your browser at **http://localhost:8501**

---

## 🧠 How LIME Works
LIME (Local Interpretable Model-Agnostic Explanations) explains 
predictions by perturbing input text and observing prediction changes.

🟢 **Green bars** → words pushing toward REAL  
🔴 **Red bars** → words pushing toward FAKE

---

## 📊 Dataset
**ISOT Fake News Dataset**
- ~21,000 real news articles from Reuters
- ~23,000 fake news articles from unreliable sources
- 44,000+ labeled articles total

---

## 🛠️ Tech Stack
- **Streamlit** — web application
- **scikit-learn** — ML classifiers and TF-IDF
- **LIME** — explainability
- **matplotlib** — visualization
- **pandas / numpy** — data processing
- **joblib** — model serialization

---

## 📜 License
This project is for academic purposes only.

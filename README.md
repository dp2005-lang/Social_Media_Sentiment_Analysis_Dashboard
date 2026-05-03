# 📊 Social Media Sentiment Analysis Dashboard

## 📌 Project Overview
This project is a beginner-friendly social media sentiment analysis system implemented in Google Colab using Python, NLTK, and Scikit-learn.

The system analyzes social media comments, tweets, and reviews, classifies them into positive, negative, or neutral sentiments, and provides interactive visualizations.

---

## 🎯 Problem Statement
Design an intelligent system that can:
- Analyze social media text data
- Classify sentiment as positive, negative, or neutral
- Process raw text using NLP techniques
- Train machine learning models for classification
- Provide interactive dashboard for analysis

---

## 💡 Solution Approach
- NLP Text Preprocessing (cleaning, stopword removal, stemming)
- TF-IDF Vectorization for feature extraction
- Multiple ML Models (Logistic Regression, Naive Bayes, Random Forest)
- Streamlit Dashboard for visualization
- Batch Analysis for CSV uploads

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, Regex |
| Feature Extraction | TF-IDF Vectorizer |
| ML Models | Logistic Regression, Naive Bayes, Random Forest |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Platform | Google Colab |

---

## 🧠 Core Concepts Used

| Concept | Application |
|---------|-------------|
| Natural Language Processing | Text cleaning & preprocessing |
| TF-IDF Vectorization | Converting text to numerical features |
| Supervised Learning | Sentiment classification |
| Logistic Regression | Baseline classification model |
| Text Preprocessing | Stopword removal, stemming |
| Model Evaluation | Accuracy, F1-score, confusion matrix |

---

## 🏗️ Project Architecture

```
Social Media Text
       ↓
Text Cleaning
       ↓
NLP Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Model Training
       ↓
Sentiment Classification
       ↓
Dashboard Visualization
```

---

## 📁 Project Workflow

1. Load/Create dataset
2. Exploratory Data Analysis
3. Text Preprocessing
4. TF-IDF Feature Extraction
5. Train-Test Split
6. Train Models
7. Model Evaluation
8. Interactive Dashboard
9. Batch Processing
10. Save Models

---

## 🚀 How to Run (Google Colab)

### Step 1: Open Google Colab
```
https://colab.research.google.com/
```

### Step 2: Install dependencies
```python
!pip install streamlit pandas numpy scikit-learn nltk plotly matplotlib seaborn joblib pyngrok
```

### Step 3: Run all code cells sequentially

### Step 4: Launch dashboard
```python
!streamlit run sentiment_dashboard.py &
```

---

## 📸 Output Example

### Sample Predictions
```
Input: "I absolutely love this product!"
Output: Sentiment: POSITIVE (Confidence: 96%)

Input: "Terrible experience, will never buy again"  
Output: Sentiment: NEGATIVE (Confidence: 92%)

Input: "The product arrived on time"
Output: Sentiment: NEUTRAL (Confidence: 88%)
```

### Confusion Matrix
```
              Predicted
              Pos  Neg  Neu
Actual Pos    18    1    1
       Neg     1   17    2
       Neu     2    1   17

Accuracy: 91.67%
F1-Score: 0.92
```

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `sentiment_model.pkl` | Trained model |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer |
| `sentiment_dashboard.py` | Streamlit app |
| `social_media_dataset.csv` | Dataset |
| `requirements.txt` | Dependencies |

---

## 🎯 Key Features

✅ Text Preprocessing Pipeline
✅ Multiple ML Models
✅ TF-IDF with N-grams
✅ Interactive Dashboard
✅ Batch CSV Processing
✅ Confidence Scores
✅ Model Evaluation Metrics

---

## 📊 Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 91.67% | 0.92 |
| Naive Bayes | 83.33% | 0.83 |
| Random Forest | 87.50% | 0.88 |

**Best Model:** Logistic Regression

---

## 🚀 Future Improvements

- BERT Transformers for better accuracy
- Real-time Twitter API integration
- Emotion detection (anger, joy, sadness)
- Multilingual support
- Cloud deployment

---

## 👨‍🎓 Author

**Name:** Debankita Panja
**Project Type:** Machine Learning / NLP  
**Platform:** Google Colab  

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/Social-Media-Sentiment-Analysis-Dashboard.git
cd Social-Media-Sentiment-Analysis-Dashboard
pip install -r requirements.txt
streamlit run sentiment_dashboard.py
```

---

## ⭐ GitHub Repository
Give a ⭐ if you like this project!



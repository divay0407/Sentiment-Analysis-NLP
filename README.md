# 💬 Sentiment Analysis (NLP)

A machine learning project that classifies text reviews as **Positive**, **Neutral**, or **Negative** using Natural Language Processing (NLP) techniques, built on a multi-platform review dataset.

---

## 📌 Overview

Understanding customer sentiment at scale is critical for businesses monitoring feedback across multiple platforms. This project builds a complete NLP pipeline — from raw text cleaning to a trained classification model — that automatically predicts the sentiment behind any given review or piece of text.

---

## 📊 Dataset

- **Columns:** `id`, `text`, `sentiment`, `source`, `date`
- **Target Variable:** `sentiment` (Negative → 0, Neutral → 1, Positive → 2)
- **Source column:** Reviews collected across different platforms, enabling cross-platform sentiment comparison
- Source: *[Add dataset source/link here]*

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn
- **NLP Tools:** Regex (re), NLTK (stopwords, WordNetLemmatizer, POS tagging), TF-IDF Vectorization
- **Environment:** Google Colab / Jupyter Notebook

---

## 🔍 Project Workflow

### 1. Data Loading & Exploration
- Loaded `sentiments.csv` into a Pandas DataFrame
- Checked shape, column types, missing values, and class distribution across `sentiment`

### 2. Data Cleaning
- Dropped null values and duplicate reviews
- Verified and standardized sentiment label formatting

### 3. Label Encoding
- Mapped sentiment labels to numeric form:
  `negative` → 0, `neutral` → 1, `positive` → 2

### 4. Text Preprocessing (NLP Pipeline)
- Lowercased all text
- Removed URLs, emails, numbers, and punctuation using **Regex**
- Removed stopwords using **NLTK**
- Applied **POS-tagged Lemmatization** (using `WordNetLemmatizer` + `pos_tag`) for more accurate root-word extraction than basic stemming

### 5. Feature Extraction (Vectorization)
- Converted cleaned text into numeric features using **TF-IDF Vectorization**
- Experimented with unigrams and bigrams (`ngram_range`) to capture short phrases

### 6. Model Building
- Trained and compared multiple classification models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- Selected the best-performing model based on weighted F1-score

### 7. Model Evaluation
- Evaluated using Accuracy, Precision, Recall, F1-score (weighted, for multi-class)
- Visualized results with a Confusion Matrix
- Cross-validated for score stability

### 8. Custom Prediction Function
- Built a reusable function to clean, vectorize, and predict sentiment for any new input sentence

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | *100%* |
| Precision (weighted) | *1.00* |
| Recall (weighted) | *1.00* |
| F1-Score (weighted) | *1.00* |

*[Multinomial Naive Bayes Model Works the Best]*

### Observations
- The **Neutral** class is generally the hardest to classify correctly, as is common in sentiment analysis
- TF-IDF with unigrams struggles with subtle/context-dependent sentences (e.g., "Great in theory, terrible in practice") — this is a known limitation of Bag-of-Words style approaches
- Adding bigrams (`ngram_range=(1,2)`) improved handling of short sentiment-bearing phrases

---

## 🚀 How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/divay0407/sentiment-analysis-nlp.git
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn
   ```
3. Open the notebook in Jupyter or Google Colab and run all cells sequentially

### Predict Sentiment for Your Own Text
```python
predict_sentiment("This app crashes every time I open it, so frustrating.")
# Output: Negative
```

---

## 📂 Repository Structure

```
sentiment-analysis-nlp/
│
├── sentiment_analysis.ipynb     # Main notebook with full pipeline
├── cleaned_sentiments.csv       # Cleaned & preprocessed dataset
├── sentiment_model.pkl          # Trained classification model
├── sentiment_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── README.md                    # Project documentation
└── sentiments.csv                # (Optional) Raw dataset
```

---

## 🔮 Future Improvements

- Address class imbalance using class weighting or oversampling (SMOTE)
- Expand vocabulary coverage (tune `max_features`, `min_df` in TF-IDF)
- Try n-gram ranges beyond bigrams for better phrase-level context
- Explore pretrained models (VADER for rule-based sentiment, or Hugging Face Transformers like BERT) for better handling of context, negation, and sarcasm
- Compare sentiment trends across different `source` platforms
- Deploy as an interactive Streamlit app for real-time sentiment prediction

---

## 👤 Author

**Divay**
B.Tech Electrical Engineering | Proficient in Data Science AI / ML Enthusiast

---

*Feel free to explore, raise issues, or suggest improvements!*

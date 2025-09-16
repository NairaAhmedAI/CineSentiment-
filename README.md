# 🎬 Sentiment Analysis on IMDB Movie Reviews  

## 📌 Overview  
This project focuses on **Sentiment Analysis** of IMDB movie reviews, a classic **Natural Language Processing (NLP)** task.  
The main objective is to classify each review as either **Positive** or **Negative** by building a complete pipeline that covers:  

- Data Collection  
- Text Preprocessing  
- Machine Learning & Deep Learning Models (next steps)  

By leveraging NLP techniques, we aim to transform raw unstructured text into clean and meaningful representations that can be used for predictive modeling.  

---

## 📊 Dataset  
We used the **IMDB Dataset of 50K Movie Reviews**, which contains **25,000 positive** and **25,000 negative** reviews.  
This dataset is widely used for benchmarking sentiment classification models.  

📥 Dataset link: [IMDB 50K Movie Reviews - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

---

## 🛠 Data Preprocessing  

Preprocessing is a critical step in NLP as raw text is often noisy and inconsistent. We applied several transformations to prepare the dataset:  

### 1️⃣ Lowercasing  
All reviews were converted to lowercase to maintain consistency and avoid duplication caused by case sensitivity.  

### 2️⃣ Label Encoding  
The target variable (**sentiment**) was encoded into numeric values:  
- `0` → Negative  
- `1` → Positive  

### 3️⃣ Cleaning (Regex-based)  
- Removed **emojis** and **hashtags**  
- Removed HTML tags such as `<br>`  
- Normalized text by stripping unnecessary characters  

### 4️⃣ Tokenization & Stopword Removal  
Using **NLTK**, we split sentences into individual tokens (words).  
Stopwords (e.g., *is, the, and*) were removed to reduce noise, but **negation words** (e.g., *not, no*) were kept as they strongly influence sentiment.  

### 5️⃣ Lemmatization with POS Tagging  
- Applied **Part-of-Speech (POS) tagging** to identify word categories (noun, verb, adjective, adverb).  
- Used **WordNet Lemmatizer** from NLTK to reduce words to their base form (e.g., *running → run*, *better → good*).  

This step ensures that semantically similar words are treated the same way, improving model generalization.  

---

## 🔑 Why NLTK?  
The **Natural Language Toolkit (NLTK)** is a powerful and widely-used Python library for NLP.  
It provides tools for:  
- Tokenization  
- Stopword filtering  
- POS tagging  
- Lemmatization  
- Corpora and lexical resources (WordNet)  

Its flexibility and ease of use make it an excellent choice for preprocessing in text classification projects like this one.  

---

📌 *Next steps: Feature extraction (TF-IDF, Word Embeddings) and building Machine Learning & Deep Learning models.*  


🔹 Logistic Regression → Balanced, robust & interpretable

🔹 SVM → Best trade-off between accuracy & generalization → ⭐ Recommended model

🔹 Random Forest → Useful for feature importance & non-linear patterns

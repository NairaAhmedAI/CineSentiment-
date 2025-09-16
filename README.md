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

# 🤖 Machine Learning Models

After preprocessing the IMDB dataset, we trained several **classical machine learning models** to classify movie reviews as **positive** or **negative**.  
Each model was trained using **TF-IDF** or **Bag-of-Words** representations, then evaluated on the test set.

---

## 🛠️ Tools & Libraries
| Category            | Libraries                                                                 |
|---------------------|----------------------------------------------------------------------------|
| Data Handling       | pandas, numpy                                                            |
| Visualization       | matplotlib, seaborn                                                        |
| ML Models           | scikit-learn (Logistic Regression, Naive Bayes, SVM, Random Forest)        |
| NLP Preprocessing   | nltk (tokenization, stopwords removal, lemmatization, POS tagging, etc.)   |

---

## 📊 Model Training & Results  

| Model                           | Accuracy  | Key Points                                                                 |
|---------------------------------|-----------|----------------------------------------------------------------------------|
| Model                           | Accuracy  | Key Points                                                                 |
|---------------------------------|-----------|----------------------------------------------------------------------------|
| **Logistic Regression (TF-IDF)** 🎯 | **89.11%** | - Balanced & interpretable <br> - Robust for text classification <br> - Implemented a **Custom Threshold = 0.6** instead of the default 0.5. <br> - This adjustment allowed the model to classify reviews into **three categories**: <br> • **Positive** (high confidence in positive sentiment) <br> • **Negative** (high confidence in negative sentiment) <br> • **Neutral** (when neither class exceeds the threshold, capturing uncertain cases) <br> - Confusion Matrix:<br> • TN: 4342 • TP: 4569 <br> • FP: 619 • FN: 470 |

| **Naive Bayes**                 | 85.27%    | - Lightweight & fast <br> - Solid baseline model <br> - Works best on simpler datasets |
| **Support Vector Machine (SVM)** ⭐ | **89.38%** | - Handles high-dimensional text data <br> - Strong generalization ability <br> - Best trade-off for real-world use |
| **Random Forest**               | 86.12%    | - Captures non-linear relationships <br> - Reduces overfitting <br> - Provides feature importance |


---

## 📊 Model Comparison Chart

| Model                | Logistic Regression | Naive Bayes | SVM   | Random Forest |
|----------------------|---------------------|-------------|-------|---------------|
| **Accuracy (%)**     | 89.11               | 85.27       | 89.38 | 86.12         |

📌 **SVM achieved the highest accuracy (89.38%) and is considered the most suitable model for deployment.**

---

## 📈 Visualization

- **Confusion Matrices** were plotted for each model to analyze classification errors.  
- **Accuracy Comparison Bar Chart** clearly shows SVM and Logistic Regression outperforming other models.  

![Model Comparison Chart](path_to_your_chart.png)

---

✅ Next step: Extend to **Deep Learning models** (LSTM, Bi-LSTM, CNN, Transformers).

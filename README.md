🎬 CineSentiment - Sentiment Analysis on IMDB Reviews
📖 Overview

CineSentiment 🎥 is a Natural Language Processing (NLP) project for sentiment analysis of IMDB movie reviews.
The project classifies reviews into Positive, Negative, or Neutral using TF-IDF feature extraction and multiple Machine Learning models.

🛠️ Tools & Libraries
Category	Libraries
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
ML Models	scikit-learn (Logistic Regression, Naive Bayes, SVM, Random Forest)
NLP Preprocess	nltk (stopwords removal, tokenization, etc.)
⚙️ Preprocessing Steps

✅ Convert text to lowercase

✅ Remove punctuation, numbers, special characters

✅ Remove stopwords

✅ Apply tokenization

✅ Extract features with TF-IDF Vectorizer

📊 Model Training & Results
Model	Accuracy	Key Points
Logistic Regression (TF-IDF)	89.11% 🎯	- Balanced & interpretable
- Robust for text classification
- Confusion Matrix:
• TN: 4342 • TP: 4569
• FP: 619 • FN: 470
Naive Bayes	85.27%	- Lightweight & fast
- Solid baseline model
- Works best on simpler datasets
Support Vector Machine (SVM)	89.38% ⭐	- Handles high-dimensional text data
- Strong generalization ability
- Best trade-off for real-world use
Random Forest	86.12%	- Captures non-linear relationships
- Reduces overfitting
- Provides feature importance
📊 Model Comparison Chart
Model	Logistic Regression	Naive Bayes	SVM	Random Forest
Accuracy (%)	89.11	85.27	89.38	86.12

📌 SVM achieved the highest accuracy (89.38%) and is the most suitable model for deployment.

📈 Visualizations

☁️ WordCloud → Highlights most frequent words

📊 Bar Charts → Show dataset distribution & balance

📉 Accuracy Plot → Compare models side by side

✅ Conclusion

🔹 Naive Bayes → Good baseline, fast & lightweight

🔹 Logistic Regression → Balanced, robust & interpretable

🔹 SVM → Best trade-off between accuracy & generalization → ⭐ Recommended model

🔹 Random Forest → Useful for feature importance & non-linear patterns

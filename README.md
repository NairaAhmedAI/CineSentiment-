
🎬 CineSentiment - Sentiment Analysis on IMDB Reviews
📖 Overview

CineSentiment 🎥 is a Natural Language Processing (NLP) project for sentiment analysis of IMDB movie reviews.
The goal is to classify reviews into Positive, Negative, or Neutral sentiments using text preprocessing, TF-IDF feature extraction, and Machine Learning models.

The project includes:

Data cleaning and preprocessing

Feature extraction with TF-IDF Vectorizer

Training and evaluation of multiple ML models

Visualizations (WordClouds, bar charts, and comparison plots)

🛠️ Tools & Libraries

pandas, numpy → Data handling and numerical operations

matplotlib, seaborn → Data visualization

scikit-learn (sklearn) → TF-IDF, Logistic Regression, Naive Bayes, SVM, Random Forest

nltk → Text preprocessing (stopwords removal, tokenization, etc.)

⚙️ Preprocessing Steps

✔️ Convert text to lowercase
✔️ Remove punctuation, numbers, and special characters
✔️ Remove stopwords
✔️ Apply tokenization
✔️ Extract features using TF-IDF Vectorizer

📊 Model Training & Results

After preprocessing and applying TF-IDF, four machine learning models were trained and compared:

🔹 Logistic Regression (TF-IDF)

Accuracy: 89.11% 🎯

Strength: Balanced performance, interpretable, robust for text classification

Confusion Matrix:

True Negatives: 4342

True Positives: 4569

False Positives: 619

False Negatives: 470

🔹 Naive Bayes

Accuracy: 85.27%

Strength: Lightweight and fast, provides a good baseline model

Works best on smaller text datasets but less robust on complex text

🔹 Support Vector Machine (SVM)

Accuracy: 89.38%

Strength: Handles high-dimensional data effectively, strong generalization ability

Best trade-off between accuracy and generalization → most suitable for real-world use

🔹 Random Forest

Accuracy: 86.12%

Strength: Reduces overfitting, captures non-linear relationships

Provides feature importance insights

📈 Model Comparison
Model	Accuracy	Strength
Logistic Regression	89.11%	Balanced, interpretable
Naive Bayes	85.27%	Lightweight, fast baseline
SVM	89.38%	Best trade-off, strong generalization
Random Forest	86.12%	Non-linear patterns, feature importance

📌 From the results, SVM showed the highest accuracy (89.38%) and strong generalization, making it the best candidate for deployment.

📊 Visualizations

WordCloud → Highlights most frequent words in the dataset

Bar Charts → Distribution of classes and text statistics

Accuracy Comparison Plot → Side-by-side performance of models

✅ Conclusion

Naive Bayes → good baseline

Logistic Regression → robust and interpretable

SVM → best trade-off between accuracy & generalization (most recommended)

Random Forest → useful for interpretability and non-linear patterns

‎# 📌 NLP Project - Logistic Regression with TF-IDF


‎## 🛠️ Imports & Preprocessing

‎- **Imported Libraries**:  ‎
‎  - `pandas`, `numpy` → Data manipulation and numerical operations  ‎
‎  - `matplotlib`, `seaborn` → Data visualization and charts  ‎
‎  - `sklearn` → Machine learning (TF-IDF, Logistic Regression, ‎evaluation metrics)  ‎
‎  - `nltk` → Text preprocessing (stopwords removal, tokenization, ‎etc.)  ‎

‎- **Preprocessing Steps**:  ‎
‎  - Converted text to lowercase  ‎
‎  - Removed punctuation, numbers, and irrelevant symbols  ‎
‎  - Removed stopwords for cleaner representation  ‎
‎  - Applied tokenization for splitting text into words  ‎
‎  - Prepared the cleaned data for TF-IDF feature extraction  ‎

‎---‎


‎## 📖 Overview
This project applies **text preprocessing** and **machine learning** ‎techniques to classify text using **Logistic Regression** with **TF-‎IDF features**.  ‎
The workflow includes:‎
‎- Data cleaning and preprocessing  ‎
‎- Feature extraction with TF-IDF  ‎
‎- Logistic Regression model training  ‎
‎- Evaluation using accuracy and confusion matrix  ‎
‎- Visualization with WordCloud and bar charts  ‎

‎---‎

‎## 📊 Model Performance

‎- **Accuracy Achieved**: **89.11%** 🎯

‎### ✅ Confusion Matrix (Logistic Regression - TF-IDF)‎

‎|                | Predicted Negative | Predicted Positive |‎
‎|----------------|--------------------|--------------------|‎
‎| **Actual Negative** | **4342** (True Negative) | **619** (False ‎Positive) |‎
‎| **Actual Positive** | **470** (False Negative) | **4569** (True ‎Positive) |‎

🔹 **Interpretation**:  ‎
‎- The model correctly predicted **4342 negatives** and **4569 ‎positives**.  ‎
‎- **619 samples** were misclassified as Positive (False Positives).  ‎
‎- **470 samples** were misclassified as Negative (False Negatives).  ‎

‎---‎

‎## 📈 Visualizations

‎### ☁️ Word Cloud
The Word Cloud highlights the most frequent words in the dataset after ‎preprocessing.  ‎
It provides an intuitive view of the **key terms** that dominate the ‎corpus, which helps in understanding the text distribution.  ‎

‎### 📊 Bar Charts
The bar charts visualize the **distribution of labels** and other ‎dataset characteristics.  ‎
They help check for **class balance** and general text statistics.‎

‎---‎

‎## 🔎 TF-IDF Explanation

‎**TF-IDF** = **Term Frequency – Inverse Document Frequency**  ‎

‎- **TF (Term Frequency):** How often a word appears in a document.  ‎
‎- **IDF (Inverse Document Frequency):** How unique or rare the word is ‎across all documents.  ‎
‎- Final TF-IDF score = **TF × IDF**, giving higher weight to ‎important, distinguishing words.  ‎

👉 This ensures that common words like *"the"*, *"and"* get **lower ‎weight**, while unique, informative words get **higher importance**.‎

‎---‎

‎## ➕ Custom Threshold & Neutral Class

By default, Logistic Regression predicts only **Positive** or ‎‎**Negative**.  ‎
A **custom threshold strategy** was added to introduce a **third ‎class: Neutral**.  ‎

‎### 🔹 How it works:‎
‎- If the probability for Positive ≥ 0.6 → classify as **Positive**  ‎
‎- If the probability for Negative ≥ 0.6 → classify as **Negative**  ‎
‎- Otherwise → classify as **Neutral**  ‎

‎### 🌟 Benefit:‎
This prevents forcing uncertain cases into Positive/Negative and ‎instead assigns them as **Neutral**.  ‎
It is especially valuable in **Sentiment Analysis**, where some text ‎may not strongly express either polarity.‎

‎---‎


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

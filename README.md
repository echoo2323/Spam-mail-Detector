# 📧 Spam Mail Detector

A Machine Learning project that classifies mail as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

---

## 📌 Project Overview

The Spam Mail Detector analyzes mail and predicts whether they are spam or legitimate (ham). The project uses text preprocessing, TF-IDF vectorization, and a Machine Learning model to achieve accurate classification.

---

## 🚀 Features

- Load and preprocess SMS dataset
- Clean text data using NLP
- Remove stopwords
- Convert text into numerical features using TF-IDF
- Train a Multinomial Naive Bayes model
- Predict custom SMS messages
- Save the trained model using Joblib

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
Spam-Mail-Detector/
│
├── dataset/
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Explore Dataset
3. Text Preprocessing
4. Remove Stopwords
5. TF-IDF Vectorization
6. Train/Test Split
7. Train Multinomial Naive Bayes Model
8. Evaluate Accuracy
9. Save Model
10. Predict Custom Messages

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/echoo2323/Spam-Mail-Detector.git
```

### Move to the project folder

```bash
cd Spam-Mail-Detector
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python app.py
```

---

## 📈 Sample Prediction

Input:

```
Congratulations! You have won a FREE iPhone. Click here to claim your prize.
```

Output:

```
Prediction: SPAM
```

---

## 👨‍💻 Author

**Pranav Kumar**

GitHub: https://github.com/echoo2323

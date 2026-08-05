import pandas as pd
import sys
print(sys.executable)
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib   

nltk.download('stopwords')

df = pd.read_csv(
    "dataset/spam.csv",
    sep="\t",
    names=["label", "message"]
)

print("Dataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nSpam / Ham Count:")
print(df["label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

stop_words = set(stopwords.words("english"))

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words
    return " ".join(words)

# Create a cleaned message column
df["clean_message"] = df["message"].apply(clean_text)

print("\nOriginal Message:\n")
print(df.loc[2, "message"])

print("\nCleaned Message:\n")
print(df.loc[2, "clean_message"])


# STEP 4 - TF-IDF Vectorization

# Convert text into numbers

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df["clean_message"])

y = df["label"]

print("\nTF-IDF Shape:")
print(X.shape)

print("\nLabels:")
print(y.head())


# STEP 5 - Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# STEP 6 - Train Model

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# STEP 7 - Model Evaluation


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# STEP 8 - Save Model


joblib.dump(model, "model/spam_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")


# STEP 9 - Predict New Message


def predict_spam(message):

    cleaned = clean_text(message)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    return prediction[0]


print("\n==============================")
print("SPAM MAIL DETECTOR")
print("==============================")

message = input("\nEnter your message:\n")

result = predict_spam(message)

if result == "spam":
    print("\nPrediction : SPAM")
else:
    print("\nPrediction : HAM (Not Spam)")
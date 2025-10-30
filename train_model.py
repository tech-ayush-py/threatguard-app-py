# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("--- Starting model training ---")

# Load the dataset
data = pd.read_csv('data.csv')
data.dropna(inplace=True)
X = data['URL']
y = data['Label']

# Create and train the vectorizer
url_vectorizer = TfidfVectorizer()
X_tfidf = url_vectorizer.fit_transform(X)

# Create and train the model
url_model = LogisticRegression()
url_model.fit(X_tfidf, y)

# --- Save the trained objects to files ---
joblib.dump(url_vectorizer, 'url_vectorizer.pkl')
joblib.dump(url_model, 'url_model.pkl')

print("--- Model and vectorizer saved to .pkl files! ---")
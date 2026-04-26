import streamlit as st
import nltk
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="🎬 Sencine", layout="centered")

st.title("🎬 Sencine - Analyse de sentiments")

# =========================
# NLTK SETUP
# =========================
@st.cache_resource
def load_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

load_nltk()

# =========================
# NLP
# =========================
stop_words = set(stopwords.words('english') + stopwords.words('french'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# =========================
# MODEL TRAINING + TUNING
# =========================
@st.cache_data
def train_model():
    df = pd.read_csv("data.csv")

    df["label"] = df["label"].astype(str).str.lower().str.strip()

    df["label"] = df["label"].map({
        "positive": 1,
        "positif": 1,
        "1": 1,
        "negative": 0,
        "negatif": 0,
        "0": 0
    })

    df = df.dropna()

    df["clean_text"] = df["text"].apply(preprocess)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    # 🔵 Naive Bayes tuning
    nb = MultinomialNB()
    nb_grid = GridSearchCV(nb, {"alpha": [0.1, 0.5, 1, 2]}, cv=3)
    nb_grid.fit(X, y)
    nb_model = nb_grid.best_estimator_

    # 🌳 Decision Tree tuning
    dt = DecisionTreeClassifier()
    dt_grid = GridSearchCV(dt, {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }, cv=3)
    dt_grid.fit(X, y)
    dt_model = dt_grid.best_estimator_

    return nb_model, dt_model, vectorizer

nb_model, dt_model, vectorizer = train_model()

# =========================
# MODELS CHOICE
# =========================
model_choice = st.radio("🤖 Choisir modèle", ["Naive Bayes", "Decision Tree"])

# =========================
# PREDICTION
# =========================
def predict(text, model_choice):
    clean = preprocess(text)
    vect = vectorizer.transform([clean])

    model = nb_model if model_choice == "Naive Bayes" else dt_model

    pred = model.predict(vect)[0]

    try:
        proba = model.predict_proba(vect)[0]
        confidence = max(proba)
    except:
        confidence = 1.0

    if confidence < 0.6:
        return "🤔 Avis pas clair", confidence

    return ("😊 Positif" if pred == 1 else "😡 Négatif"), confidence

# =========================
# APPLICATION
# =========================
st.subheader("💬 Tester un avis")

text = st.text_area("Écris ton avis ici")

if st.button("Analyser"):
    if text:
        result, conf = predict(text, model_choice)
        st.success(result)
        st.write("Confiance :", round(conf, 2))

# =========================
# EXPPLICATION (UNE SEULE PAGE)
# =========================
st.markdown("---")
st.title("📘 Explication du projet")

st.markdown("""
## 🧠 Phase 1 : NLP
- Tokenization
- Stopwords removal
- Lemmatization
- Stemming  
👉 Nettoyage du texte

---

## 🤖 Phase 2 : Modèles
- Naive Bayes
- Decision Tree  
👉 Classification des sentiments

---

## 🔧 Phase 3 : Tuning
- GridSearchCV
- Optimisation automatique des paramètres  
👉 Améliorer la précision

---

## 📊 Phase 4 : Performance
- Évaluation des modèles
- Analyse des résultats
👉 Vérifier la fiabilité du système

---

## 🎯 Conclusion
Ce projet permet de classifier automatiquement les avis grâce au NLP et au Machine Learning.
""")

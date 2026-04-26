import streamlit as st
import nltk
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# CONFIG

st.set_page_config(page_title="🎬 Sencine", layout="centered")


# NLTK setup

@st.cache_resource
def load_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

load_nltk()


# NLP Tools

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


#  Model

@st.cache_data
def train_model():
    try:
        df = pd.read_csv("data.csv")
        df["clean_text"] = df["text"].apply(preprocess)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["clean_text"])
        y = df["label"]

        model = MultinomialNB()
        model.fit(X, y)

        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Fichier **data.csv** introuvable.")
        st.stop()

model, vectorizer = train_model()


# Prediction

def predict(text):
    clean = preprocess(text)
    vect = vectorizer.transform([clean])

    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0]

    confidence = max(proba)  # niveau de confiance

    # Seuil de confiance (tu peux ajuster)
    if confidence < 0.6:
        return "🤔 Veuillez donner un avis plus clair !",  clean, confidence

    if pred == 1:
        result = "😊 Positif"
    else:
        result = "😡 Négatif"

    return result, clean, confidence

# -----------------------------
# LOAD MOVIES (nouveau fichier CSV)
# -----------------------------

st.title("🎬 Analyser les films et les series sénégalaises") 

try:
    movies_df = pd.read_csv("movies.csv")
    
    # Nettoyage important (supprime les espaces en trop)
    movies_df["titre"] = movies_df["titre"].astype(str).str.strip()
    
    # Liste triée pour le selectbox
    movie_list = sorted(movies_df["titre"].unique().tolist())
    
except FileNotFoundError:
    st.error("❌ Fichier **movies.csv** introuvable. Place-le dans le même dossier que ton app.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement du CSV : {e}")
    st.stop()

st.subheader("🎬 Choisissez un film ou un série")
movie_selected = st.selectbox("Liste des films et séries sénégalaises", movie_list)

# Récupération propre de la ligne
selected_row = movies_df[movies_df["titre"] == movie_selected].iloc[0]

# Affichage enrichi
st.write(f"**Type :** {selected_row['type']}")
st.write(f"**Année :** {selected_row.get('annee', 'Non renseignée')}")
st.write(f"**Réalisateur :** {selected_row.get('realisateur', 'Non renseigné')}")
st.write("📖 **Description :**")
st.write(selected_row["description"])


# User Review

st.subheader("💬 Donnez votre avis")
user_review = st.text_area("Votre impression sur le film", height=150)

if st.button("🔍 Valider mon avis", type="primary"):
    if not user_review.strip():
        st.warning("Veuillez écrire un avis.")
    else:
        result, clean_text, confidence = predict(user_review)

        st.success(f"**Sentiment détecté :** {result}")
        st.write(f"Confiance : {confidence:.2f}")

        with st.expander("Texte nettoyé"):
            st.code(clean_text)

        # ❌ NE PAS SAUVEGARDER SI PAS CLAIR
        if confidence < 0.6:
            st.warning("⚠️ Avis non enregistré car il n’est pas assez clair.")
        else:
            new_data = pd.DataFrame({
                "film": [movie_selected],
                "review": [user_review],
                "sentiment": [result]
            })

            file_path = "reviews.csv"
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                new_data.to_csv(file_path, index=False)
            else:
                new_data.to_csv(file_path, mode='a', header=False, index=False)

            st.success("✅ Avis sauvegardé !")
            st.rerun()
# -----------------------------
#Enregistrement 

st.subheader("📊 Avis sauvegardés")

def load_reviews():
    file_path = "reviews.csv"
    expected_columns = ["film", "review", "sentiment"]

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame(columns=expected_columns)

    try:
        df = pd.read_csv(file_path)

        # Si le fichier n'a pas les bonnes colonnes → on le réinitialise
        if not all(col in df.columns for col in expected_columns):
            st.warning("Le fichier reviews.csv est mal formé. Il va être réinitialisé.")
            return pd.DataFrame(columns=expected_columns)

        df = df[expected_columns]                    # garder seulement les colonnes utiles
        df = df.dropna(how='all').reset_index(drop=True)
        return df

    except Exception:
        return pd.DataFrame(columns=expected_columns)
df_reviews = load_reviews()

if df_reviews.empty:
    st.info("Aucun avis sauvegardé pour le moment.")
else:
    # Filtre par film
    unique_films = sorted(df_reviews["film"].dropna().unique().tolist())
    film_filter = st.selectbox(
        "Filtrer par film",
        ["Tous"] + unique_films
    )

    if film_filter != "Tous":
        filtered_reviews = df_reviews[df_reviews["film"] == film_filter].copy()
    else:
        filtered_reviews = df_reviews.copy()

    st.dataframe(filtered_reviews, use_container_width=True)

    # Statistiques
    st.subheader("📈 Statistiques")
    total = len(filtered_reviews)
    positives = (filtered_reviews["sentiment"].astype(str).str.contains("Positif", na=False)).sum()
    negatives = total - positives

    col1, col2, col3 = st.columns(3)
    col1.metric("Total avis", total)
    col2.metric("😊 Positifs", positives)
    col3.metric("😡 Négatifs", negatives)

    if total > 0:
        st.progress(positives / total, text=f"Pourcentage positif : {positives/total:.1%}")


# Sidebar

st.sidebar.title("📘 À propos")
st.sidebar.info("App d'analyse de sentiment sur les avis de films\n\nTechnologies : Streamlit + NLTK + Naive Bayes")

if st.sidebar.button("🗑️ Effacer tous les avis"):
    if os.path.exists("reviews.csv"):
        os.remove("reviews.csv")
        st.success("Tous les avis ont été supprimés.")
        st.rerun()

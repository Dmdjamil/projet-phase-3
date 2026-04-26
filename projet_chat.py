import strimport nltk
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV  # ✅ AJOUT TUNING

# CONFIG
st.set_page_config(page_title="🎬 Sencine", layout="centered")

menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Application", "📘 Explication du projet"]
)

if menu == "🏠 Application":
    
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
    
    # =========================
    # MODELS + TUNING 🔥
    # =========================
    @st.cache_data
    def train_model():
        try:
            df = pd.read_csv("data.csv")
    
            df["label"] = df["label"].astype(str).str.lower().str.strip()
    
            df["label"] = df["label"].map({
                "positive": 1,
                "positif": 1,
                "1": 1,
                "negatif": 0,
                "negative": 0,
                "0": 0
            })
            
            df = df.dropna()
            
            df["clean_text"] = df["text"].apply(preprocess)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df["clean_text"])
            y = df["label"]
    
            # =====================
            # 🔵 NAIVE BAYES TUNING
            # =====================
            nb = MultinomialNB()
            nb_params = {
                "alpha": [0.1, 0.5, 1.0, 2.0]
            }
    
            nb_grid = GridSearchCV(nb, nb_params, cv=3, n_jobs=-1)
            nb_grid.fit(X, y)
            nb_model = nb_grid.best_estimator_
    
            # =====================
            # 🌳 DECISION TREE TUNING
            # =====================
            dt = DecisionTreeClassifier()
            dt_params = {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
    
            dt_grid = GridSearchCV(dt, dt_params, cv=3, n_jobs=-1)
            dt_grid.fit(X, y)
            dt_model = dt_grid.best_estimator_
    
            # 🔥 Affichage dans sidebar
            st.sidebar.write("🔧 NB params :", nb_grid.best_params_)
            st.sidebar.write("🔧 DT params :", dt_grid.best_params_)
            st.sidebar.write("📊 NB score :", round(nb_grid.best_score_, 3))
            st.sidebar.write("📊 DT score :", round(dt_grid.best_score_, 3))
    
            return nb_model, dt_model, vectorizer
    
        except FileNotFoundError:
            st.error("❌ Fichier **data.csv** introuvable.")
            st.stop()
    
    nb_model, dt_model, vectorizer = train_model()
    
    # =========================
    # CHOIX DU MODELE
    # =========================
    model_choice = st.radio(
        "🤖 Choisissez le modèle",
        ["Naive Bayes", "Decision Tree"]
    )
    
    # =========================
    # PREDICTION
    # =========================
    def predict(text, model_choice):
        clean = preprocess(text)
        vect = vectorizer.transform([clean])
    
        if model_choice == "Naive Bayes":
            model = nb_model
        else:
            model = dt_model
    
        pred = model.predict(vect)[0]
    
        try:
            proba = model.predict_proba(vect)[0]
            confidence = max(proba)
        except:
            confidence = 1.0
    
        if confidence < 0.6:
            return "🤔 Veuillez donner un avis plus clair !", clean, confidence
    
        if pred == 1:
            result = "😊 Positif"
        else:
            result = "😡 Négatif"
    
        return result, clean, confidence
    
    # =========================
    # MOVIES
    # =========================
    st.title("🎬 Analyser les films et les séries sénégalaises")
    
    try:
        movies_df = pd.read_csv("movies.csv")
        movies_df["titre"] = movies_df["titre"].astype(str).str.strip()
        movie_list = sorted(movies_df["titre"].unique().tolist())
    
    except FileNotFoundError:
        st.error("❌ Fichier **movies.csv** introuvable.")
        st.stop()
    
    st.subheader("🎬 Choisissez un film ou une série")
    movie_selected = st.selectbox("Liste des films et séries", movie_list)
    
    selected_row = movies_df[movies_df["titre"] == movie_selected].iloc[0]
    
    st.write(f"**Type :** {selected_row['type']}")
    st.write(f"**Année :** {selected_row.get('annee', 'Non renseignée')}")
    st.write(f"**Réalisateur :** {selected_row.get('realisateur', 'Non renseigné')}")
    st.write("📖 **Description :**")
    st.write(selected_row["description"])
    
    # =========================
    # USER REVIEW
    # =========================
    st.subheader("💬 Donnez votre avis")
    user_review = st.text_area("Votre impression", height=150)
    
    if st.button("🔍 Valider mon avis", type="primary"):
        if not user_review.strip():
            st.warning("Veuillez écrire un avis.")
        else:
            result, clean_text, confidence = predict(user_review, model_choice)
    
            st.success(f"**Sentiment :** {result}")
            st.write(f"Confiance : {confidence:.2f}")
    
            with st.expander("Texte nettoyé"):
                st.code(clean_text)
    
            if confidence < 0.6:
                st.warning("⚠️ Avis non enregistré (pas clair)")
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
    
    # =========================
    # REVIEWS DISPLAY
    # =========================
    st.subheader("📊 Avis sauvegardés")
    
    def load_reviews():
        file_path = "reviews.csv"
        cols = ["film", "review", "sentiment"]
    
        if not os.path.exists(file_path):
            return pd.DataFrame(columns=cols)
    
        try:
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in cols):
                return pd.DataFrame(columns=cols)
            return df[cols].dropna(how='all')
        except:
            return pd.DataFrame(columns=cols)
    
    df_reviews = load_reviews()
    
    if df_reviews.empty:
        st.info("Aucun avis pour le moment.")
    else:
        film_filter = st.selectbox("Filtrer", ["Tous"] + sorted(df_reviews["film"].unique()))
    
        if film_filter != "Tous":
            df_reviews = df_reviews[df_reviews["film"] == film_filter]
    
        st.dataframe(df_reviews, use_container_width=True)
    
        st.subheader("📈 Statistiques")
        total = len(df_reviews)
        positives = df_reviews["sentiment"].str.contains("Positif").sum()
        negatives = total - positives
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("😊 Positifs", positives)
        col3.metric("😡 Négatifs", negatives)
    
        if total > 0:
            st.progress(positives / total)
    
    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("📘 À propos")
    st.sidebar.info("App NLP avec Naive Bayes + Decision Tree + GridSearchCV 🔥")
    
    if st.sidebar.button("🗑️ Effacer les avis"):
        if os.path.exists("reviews.csv"):
            os.remove("reviews.csv")
            st.success("Avis supprimés")
            st.rerun()
if menu == "📘 Explication du projet":

    st.title("📘 Explication du projet Sencine")

    st.markdown("""
    ## 🧠 Phase 1 : NLP (Traitement du langage naturel)
    Le NLP permet de transformer le texte des utilisateurs en données exploitables.
    Dans ce projet, nous avons utilisé :
    - Tokenization
    - Stopwords removal
    - Lemmatization
    - Stemming

    👉 Objectif : nettoyer les avis pour les rendre compréhensibles par la machine.

    ---

    ## 🤖 Phase 2 : Modèles de classification
    Nous utilisons deux algorithmes :
    - Naive Bayes
    - Decision Tree

    👉 Objectif : prédire si un avis est positif ou négatif.

    ---

    ## 🔧 Phase 3 : Hyperparameter Tuning
    Nous utilisons GridSearchCV pour optimiser les modèles automatiquement.

    - Naive Bayes : alpha
    - Decision Tree : profondeur, critères, splits

    👉 Objectif : améliorer la précision du modèle.

    ---

    ## 📊 Phase 4 : Performance
    Cette phase permet d’évaluer le système :
    - précision des modèles
    - confiance des prédictions
    - analyse des avis utilisateurs

    👉 Objectif : vérifier la fiabilité du système.

    ---

    ## 🎬 Conclusion
    Sencine est une application d’analyse de sentiments basée sur le NLP et le machine learning.
    Elle permet de classifier automatiquement les avis des films et séries sénégalaises.
    """)

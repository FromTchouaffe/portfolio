import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# Configuration de la page
st.set_page_config(page_title="Mon Portfolio", page_icon=":briefcase:", layout="centered")

# Fonction pour charger et encoder les données
@st.cache_data
def load_and_encode_data(file_path):
    data = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    
    # Encodage des colonnes catégorielles
    data['objet__du_pret'] = label_encoder.fit_transform(data['objet__du_pret'])
    data['politique_de_credit'] = label_encoder.fit_transform(data['politique_de_credit'])
    
    return data

# Fonction pour préparer les données d'entraînement et de test
@st.cache_data
def prepare_data(data):
    X = data.drop(columns=['pret_non_remboursé'])
    y = data['pret_non_remboursé'].map({'oui': 1, 'non': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Combiner X_train et y_train pour faciliter le sur-échantillonnage
    train_data = pd.concat([X_train, y_train], axis=1)
    
    majority_class = train_data[train_data['pret_non_remboursé'] == 0]
    minority_class = train_data[train_data['pret_non_remboursé'] == 1]
    
    if not minority_class.empty:
        minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
        train_data_balanced = pd.concat([majority_class, minority_class_upsampled])
        
        X_train_balanced = train_data_balanced.drop('pret_non_remboursé', axis=1)
        y_train_balanced = train_data_balanced['pret_non_remboursé']
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    else:
        st.error("Erreur : La classe minoritaire est absente dans l'ensemble d'entraînement.")
        return None, None, None, None

# Fonction pour entraîner et évaluer le modèle
@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return conf_matrix, class_report, roc_auc

# Fonction pour afficher les images de logos
def display_logos():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    logo_paths = [
        "/Users/christiantchouaffe/Desktop/MonPortfolio/Matplotlib.png",
        "/Users/christiantchouaffe/Desktop/MonPortfolio/Numpy.png",
        "/Users/christiantchouaffe/Desktop/MonPortfolio/Pandas.png",
        "/Users/christiantchouaffe/Desktop/MonPortfolio/Plotly.png",
        "/Users/christiantchouaffe/Desktop/MonPortfolio/seaborn.png",
        "/Users/christiantchouaffe/Desktop/MonPortfolio/Sklearn.png"
    ]
    
    for col, logo in zip([col1, col2, col3, col4, col5, col6], logo_paths):
        col.image(logo, width=100)

# Fonction pour gérer l'affichage de la page d'accueil
def show_home_page():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("/Users/christiantchouaffe/Desktop/MonPortfolio/PhotoModifié.png", width=250)
        st.markdown("<h2 style='text-align: center; font-size: 24px;'>Christian Tchouaffé</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Data Analyst</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style="font-size: 16px; line-height: 1.5;">
            📞 Tel : 07 86 15 97 69<br>
            📧 Email : <a href="mailto:christiantchouaffe@orange.fr">christiantchouaffe@orange.fr</a><br>
            🌐 Réseaux : <a href="https://www.linkedin.com/in/christiantchouaffe" target="_blank">linkedin.com/in/christiantchouaffe</a><br>
            <a href="https://github.com/FromTchouaffe" target="_blank">github.com/FromTchouaffe</a>
            </p>
            """, unsafe_allow_html=True,
        )

    with col2:
        langue = st.radio("Choisissez votre langue / Choose your language", ("Français", "English"))

        if langue == "Français":
            st.markdown(
                """
                <div style="margin-left: 80px; margin-top: 20px; text-align: justify; font-size: 18px;">
                En tant que Data Analyst polyvalent, je maîtrise une large gamme d'analyses...
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="margin-left: 80px; margin-top: 20px; text-align: justify; font-size: 18px;">
                As a versatile Data Analyst, I am proficient in a wide range of analyses...
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    st.markdown(
        "<h5 style='font-size: 16px; text-align: center; margin-bottom: 5px;'>Je programme en Python et pour la réalisation des cas d'usage de ce portfolio j'ai utlisé des librairies telles que :</h5>",
        unsafe_allow_html=True
    )
    display_logos()

def show_supervised_learning_page(data, X_train_balanced, y_train_balanced, X_test, y_test):
    st.title("Apprentissage supervisé")

    st.write("**Cas d'usage :** Prédiction de la capacité d'un emprunteur à rembourser son prêt en se basant sur les données disponibles.")

    section = st.selectbox("Sélectionnez une section", ["Présentation du jeu de données", "Visualisation", "Prédiction"])
    
    if section == "Présentation du jeu de données":
        st.header("Présentation du jeu de données")
        st.write("Voici les premières lignes du dataset :")
        st.dataframe(data.head())
        info = pd.DataFrame({
            "Nom de la variable": data.columns,
            "Nombre d'occurrences": data.count(),
            "Type de variable": data.dtypes
        }).reset_index(drop=True)
        info = info.astype(str)
        st.markdown("<div style='text-align: center;'>Informations sur les variables du dataset :</div>", unsafe_allow_html=True)
        st.dataframe(info)

    elif section == "Visualisation":
        st.header("Visualisation")
        numeric_columns = data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_columns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Matrice de corrélation des variables numériques')
        st.pyplot(plt.gcf())
        
        st.header("Histogrammes des variables numériques")
        plt.figure(figsize=(15, 20))
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(5, 3, i)
            sns.histplot(data, x=column, hue='pret_non_remboursé', kde=False, multiple="stack", palette="coolwarm")
            plt.title(f'Histogramme de {column}')
            plt.xlabel(column)
            plt.ylabel('Fréquence')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
        
    elif section == "Prédiction":
        st.header("Prédiction")
        if X_train_balanced is not None and y_train_balanced is not None:
            conf_matrix, class_report, roc_auc = train_model(X_train_balanced, y_train_balanced, X_test, y_test)
            st.subheader("Résultats du Modèle")
            st.write("Matrice de confusion :")
            st.dataframe(pd.DataFrame(conf_matrix, index=['Classe 0', 'Classe 1'], columns=['Prédit 0', 'Prédit 1']))
            st.write("Rapport de classification :")
            st.dataframe(pd.DataFrame(class_report).transpose())
            st.write(f"Score AUC ROC : {roc_auc:.2f}")

# Charger les données
data = load_and_encode_data("/Users/christiantchouaffe/Desktop/MonPortfolio/loan_data_final.csv")
X_train_balanced, X_test, y_train_balanced, y_test = prepare_data(data)

page = st.sidebar.radio("Navigation", ["Accueil", "Apprentissage supervisé", "Apprentissage non supervisé", "Apprentissage profond"])

if page == "Accueil":
    show_home_page()
elif page == "Apprentissage supervisé":
    show_supervised_learning_page(data, X_train_balanced, y_train_balanced, X_test, y_test)

# Bloc "Apprentissage profond"
elif page == "Apprentissage profond":
    st.title("Big Data et Intelligence Artificielle")
    section_deep_learning = st.selectbox(
        "Sélectionnez une section", 
        ["Contexte et enjeux", "Les Large Language Models", "Cas d'usage"]
    )
    if section_deep_learning == "Contexte et enjeux":
        st.header("Contexte et Enjeux")
        st.markdown("Texte sur le contexte et les enjeux...", unsafe_allow_html=True)
    elif section_deep_learning == "Les Large Language Models":
        st.header("Les Large Language Models")
        st.markdown("Texte sur les LLMs...", unsafe_allow_html=True)
    elif section_deep_learning == "Cas d'usage":
        st.header("Cas d'usage")
        st.write("""
            <div style="text-align: justify;">
            Cette application est un chatbot dédié à accompagner les utilisateurs dans l'utilisation des logiciels de la suite Office...
            </div>
            """, unsafe_allow_html=True)

# Bloc "Cas d'usage" pour le chatbot
elif page == "Cas d'usage":
    st.header("Chatbot pour la Suite Office")

    openai_api_key = "ta_cle_api_exacte"  # Remplace par ta clé API OpenAI valide

    if not openai_api_key:
        st.error("Clé API OpenAI introuvable. Veuillez la définir.")
    else:
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = ConversationChain(llm=llm)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        def get_text():
            return st.text_input("Vous : ", "")

        user_input = get_text()

        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            try:
                output = chain.run(input=user_input)
                st.session_state["messages"].append({"role": "bot", "content": output})
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {str(e)}")

        for i, msg in enumerate(st.session_state["messages"]):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}")
            else:
                message(msg["content"], key=f"bot_{i}")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Configuration de la page
st.set_page_config(page_title="Mon Portfolio", page_icon=":briefcase:", layout="centered")


# Inject custom CSS to hide the GitHub logo and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;} /* Cache le menu principal (contenant le logo GitHub) */
    footer {visibility: hidden;} /* Cache le pied de page */
    .viewerBadge_container__1QSob {display: none;} /* Cache le logo GitHub Streamlit Cloud */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Fonction pour charger et encoder les données
@st.cache_data
def load_and_encode_data(file_path):
    # Lien vers le fichier CSV brut (raw) sur GitHub
    url = "https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/loan_data_final.csv"
    # Lecture du fichier CSV depuis l'URL
    data = pd.read_csv(url)
    
    # Renommé la variable 'pret_non_remboursé' en 'statut_prêt'
    data.rename(columns={'pret_non_remboursé': 'statut_prêt'}, inplace=True)
    
    # Modifier la valeur de la colonne 'statut_prêt' : changer 'oui' par 'non_remboursé' et 'non' par 'remboursé'
    data['statut_prêt'] = data['statut_prêt'].replace({'oui': 'non_remboursé', 'non': 'remboursé'})
    
    label_encoder = LabelEncoder()
    
    # Encodage des colonnes catégorielles
    data['objet__du_pret'] = label_encoder.fit_transform(data['objet__du_pret'])
    data['politique_de_credit'] = label_encoder.fit_transform(data['politique_de_credit'])
    
    return data

# Fonction pour préparer les données d'entraînement et de test
@st.cache_data
def prepare_data(data):
    X = data.drop(columns=['statut_prêt'])
    y = data['statut_prêt'].map({'non_remboursé': 1, 'remboursé': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Combiner X_train et y_train pour faciliter le sur-échantillonnage
    train_data = pd.concat([X_train, y_train], axis=1)
    
    majority_class = train_data[train_data['statut_prêt'] == 0]
    minority_class = train_data[train_data['statut_prêt'] == 1]
    
    if not minority_class.empty:
        minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
        train_data_balanced = pd.concat([majority_class, minority_class_upsampled])
        
        X_train_balanced = train_data_balanced.drop('statut_prêt', axis=1)
        y_train_balanced = train_data_balanced['statut_prêt']
        
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
    class_report = classification_report(y_test, y_pred, output_dict=True, target_names=['remboursé (0)', 'non_remboursé (1)'])
    roc_auc = roc_auc_score(y_test, y_pred)
    
    return conf_matrix, class_report, roc_auc


# Fonction pour afficher les images de logos
def display_logos():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    logo_paths = [
        "https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/Matplotlib.png",
        "https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/Numpy.png",
        "https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/Pandas.png",
        "https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/Plotly.png",
        "https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/seaborn.png",
        "https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/Sklearn.png"
    ]
    
    for col, logo in zip([col1, col2, col3, col4, col5, col6], logo_paths):
        col.image(logo, width=100)

# Fonction pour gérer l'affichage de la page d'accueil
def show_home_page():
    # Création des colonnes pour organiser la mise en page
    col1, col2 = st.columns([1, 2])

    with col1:
        # Ajout de la photo à gauche
        st.image("https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/PhotoModifié.png", width=250)
        # Affichage du prénom, nom (sur une seule ligne), titre et contact sous la photo
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
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Choix de la langue au-dessus du texte de présentation
        langue = st.radio("Choisissez votre langue / Choose your language", ("Français", "English"))

        # Décaler et justifier le texte de présentation avec une taille de police augmentée et le remonter pour centrer par rapport à la photo
        if langue == "Français":
            st.markdown(
                """
                <div style="margin-left: 80px; margin-top: 20px; text-align: justify; font-size: 18px;">
                <p style="font-size: 22px; font-weight: bold; text-align: justify;">Exploiter la donnée pour en extraire de la valeur.</p>
                En tant que Data Analyst polyvalent, je maîtrise une large gamme d'analyses : descriptives, hypothético-déductives, 
                inférentielles et exploratoires, avec une attention particulière à la détection de biais dans les données. 
                Mes analyses s'appuient sur des tests statistiques rigoureux et se traduisent par la rédaction de rapports détaillés, 
                la création de tableaux de bord interactifs via des outils BI comme Power BI, ou par l'automatisation de tâches à l'aide 
                de frameworks tels que Streamlit ou Voilà.
                Je réalise également des requêtes SQL sur des ERP clients et j'utilise des algorithmes de machine learning pour effectuer 
                des prédictions grâce aux techniques d'apprentissage supervisé et non supervisé. Actuellement, je m'attache à intégrer 
                la GenAI en développant une plateforme innovante, conçue pour proposer des assistants IA personnalisés aux besoins spécifiques 
                des utilisateurs.
                </div>
            """, 
            unsafe_allow_html=True
        )
        else:
            st.markdown(
                """
                <div style="margin-left: 80px; margin-top: 20px; text-align: justify; font-size: 18px;">
                <p style="font-size: 22px; font-weight: bold; text-align: justify;">Leveraging data to extract value.</p>
                As a versatile Data Analyst, I am proficient in a wide range of analyses: descriptive, hypothetico-deductive, 
                inferential, and exploratory, with a particular focus on detecting biases in data. My analyses are grounded in rigorous 
                statistical testing and result in the creation of detailed reports, interactive dashboards using BI tools such as Power BI, 
                or task automation through frameworks like Streamlit or Voilà.
                I also perform SQL queries on client ERPs and use machine learning algorithms to make predictions using both supervised 
                and unsupervised learning techniques. Currently, I am focused on integrating GenAI by developing an innovative platform 
                designed to offer AI assistants tailored to the specific needs of users.
                </div>
            """, 
            unsafe_allow_html=True
        )

    
    # Afficher les logos des bibliothèques
        st.markdown("---")
        st.markdown(
        "<h5 style='font-size: 16px; text-align: justify; margin-bottom: 5px;'>Je programme en Python et pour la réalisation des cas d'usage de ce portfolio j'ai utlisé des librairies telles que :</h5>",
        unsafe_allow_html=True
        )
        display_logos()

def show_supervised_learning_page(data, X_train_balanced, y_train_balanced, X_test, y_test):
    st.title("Apprentissage supervisé")
    st.write("**Objectif :** L'apprentissage supervisé consiste à entraîner un algorithme d'IA à estimer des valeurs futures à partir des données passées, qu'il s'agisse de biens, de services, ou de comportements.")
    # Ajout du paragraphe Cas d'usage
    st.write("**Cas d'usage :** Prédiction de la capacité d'un emprunteur à rembourser son prêt en se basant sur les données disponibles. (source du jeu de données : Kaggle.com)")

    # Menu déroulant pour la sélection de la section
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
        
        # Centrer le tableau en utilisant Streamlit
        st.markdown("<div style='text-align: center;'>Informations sur les variables du dataset :</div>", unsafe_allow_html=True)
        st.dataframe(info)

        # Ajouter le texte en dessous dans un paragraphe simple et justifié
        st.markdown(
            """
            <p style="text-align: justify;">
            Le jeu de données contient des informations détaillées sur des prêts accordés par une banque, avec un focus sur les caractéristiques financières 
            et comportementales des emprunteurs. Il inclut 14 variables, dont 13 sont explicatives et une variable cible. 
            Les variables explicatives couvrent divers aspects tels que les objectifs des prêts, les taux d'intérêt, les mensualités, 
            le revenu annuel (logarithmé), le ratio d'endettement, le score de crédit, et l'historique de crédit de l'emprunteur.
            
            La variable cible, <strong>statut_prêt</strong>, indique si le prêt a été remboursé ou non, ce qui permet d'analyser les facteurs 
            contribuant au risque de non-remboursement.

            Le jeu de données présente un déséquilibre de classes, avec moins d'exemples de prêts non remboursés. 
            Il est cependant plus important de prédire avec précision si un prêt ne sera pas remboursé que l'inverse.
            </p>
            """, 
            unsafe_allow_html=True
        )

    elif section == "Visualisation":
        st.header("Visualisation")
        
        numeric_columns = data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_columns.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Matrice de corrélation des variables numériques')
        st.pyplot(plt.gcf())
        
        # Ajouter le commentaire après les graphiques
        st.markdown(
            """
            La corrélation la plus importante est fortement négative entre le taux d'intérêt et le score de crédit. 
            Cela signifie que les individus ayant un meilleur score de crédit tendent à obtenir des taux d'intérêt plus bas, ce qui est logique dans le cadre d'une évaluation du risque par les prêteurs. 
            Il n'y a pas d'autres corrélations significatives entre les différentes variables (au-dessus de 0,7 ou en dessous de -0,7), 
            ce qui indique que les variables numériques du jeu de données sont en grande partie indépendantes les unes des autres.

            Le score de crédit pourrait être une variable particulièrement importante pour prédire si un prêt sera remboursé, car il est fortement lié à plusieurs autres variables clés (taux d'intérêt, utilisation du crédit).
            
            """
        )
        st.header("Histogrammes des variables numériques")
        plt.figure(figsize=(15, 20))
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(5, 3, i)
            sns.histplot(data, x=column, hue='statut_prêt', kde=False, multiple="stack", palette="coolwarm")
            plt.title(f'Histogramme de {column}')
            plt.xlabel(column)
            plt.ylabel('Fréquence')
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close() # Fermer le graphique pour éviter les superpositions
        
        # Ajouter le commentaire après les graphiques
        st.markdown(
            """
            Les histogrammes des variables numériques révèlent un déséquilibre entre les prêts remboursés et non remboursés.
            Cependant, les distributions au sein de chaque variable restent globalement similaires.

            
            """
        )

    elif section == "Prédiction":
        st.header("Prédiction")

        # Présentation des algorithmes après les résultats de prédiction
        st.subheader("Présentation des Algorithmes")
        st.markdown("""
            <div style="text-align: justify;">
            <ul>
            <li><strong>Random Forest</strong> : algorithme d'apprentissage supervisé qui fonctionne en construisant plusieurs arbres de décision sur différents échantillons de données. 
            Chacun de ces arbres fait des prédictions, et le Random Forest combine les prédictions des arbres pour donner une réponse finale.</li>

            <li><strong>Decision Tree</strong> : un modèle simple qui divise les données en groupes basés sur des conditions. Chaque "nœud" dans l'arbre représente une décision sur une variable, et chaque "branche" mène à une prédiction.</li>

            <li><strong>Gradient Boosting</strong> : algorithme plus avancé qui crée une série d'arbres de décision, où chaque arbre corrige les erreurs du précédent. À chaque étape, l'algorithme se "booste" pour améliorer la précision.</li>

            <li><strong>Régression Logistique</strong> : modèle statistique simple, utilisé pour prédire des résultats binaires (deux classes, comme "oui/non" ou "remboursé/non remboursé"). Contrairement à la régression linéaire, qui prédit des valeurs continues, la régression logistique prédit des probabilités.</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)


        
    # Menu déroulant pour choisir le modèle à tester
        model_choice = st.selectbox(
            "Choisissez un modèle à tester",
            ["Random Forest", "Decision Tree", "Gradient Boosting", "Régression Logistique"]
        )

    # Entraîner et afficher les résultats selon le modèle choisi
        if X_train_balanced is not None and y_train_balanced is not None:
            if model_choice == "Random Forest":
            # Entraîner le Random Forest avec les données équilibrées
                model = RandomForestClassifier(class_weight='balanced', random_state=42)
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test)
            elif model_choice == "Decision Tree":
            # Entraîner le Decision Tree avec les données équilibrées
                dt_classifier = DecisionTreeClassifier(random_state=42)
                dt_classifier.fit(X_train_balanced, y_train_balanced)
                y_pred = dt_classifier.predict(X_test)
            elif model_choice == "Gradient Boosting":
            # Entraîner le Gradient Boosting avec les données équilibrées
                gb_classifier = GradientBoostingClassifier(random_state=42)
                gb_classifier.fit(X_train_balanced, y_train_balanced)
                y_pred = gb_classifier.predict(X_test)
            elif model_choice == "Régression Logistique":
            # Entraîner la Régression Logistique avec les données équilibrées
                logreg_classifier = LogisticRegression(random_state=42, max_iter=1000)
                logreg_classifier.fit(X_train_balanced, y_train_balanced)
                y_pred = logreg_classifier.predict(X_test)

        # Générer la matrice de confusion et le rapport de classification
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True, target_names=['Remboursé', 'Non remboursé'])
            roc_auc = roc_auc_score(y_test, y_pred)

        # Afficher les résultats sous forme de tableau
            st.subheader(f"Résultats pour {model_choice}")
            st.write("Matrice de confusion :")
            conf_matrix_df = pd.DataFrame(conf_matrix, index=['Remboursé', 'Non remboursé'], columns=['Prédit Remboursé', 'Prédit Non remboursé'])
            st.dataframe(conf_matrix_df)

            st.write("Rapport de classification :")
            class_report_df = pd.DataFrame(class_report).transpose().apply(lambda x: np.round(x, 2))
            st.dataframe(class_report_df)

            st.write(f"Score AUC ROC : {roc_auc:.2f}")

        # Ajouter un commentaire spécifique pour chaque modèle
            if model_choice == "Random Forest":
                st.markdown("""
                <div style="text-align: justify;">
                <ul>
                <li><strong>Explications des métriques</strong></li>
            
                <li><strong>Précision</strong> : La précision mesure la proportion de prédictions correctes parmi toutes les prédictions faites pour une classe donnée. 
                Par exemple, une précision de 85% pour les prêts remboursés signifie que, parmi toutes les prédictions de prêts remboursés faites par le modèle, 85% étaient correctes.</li>

                <li><strong>Rappel</strong> : Le rappel, ou sensibilité, mesure la proportion de véritables cas positifs qui sont correctement identifiés par le modèle. 
                Un rappel de 98% pour les prêts remboursés signifie que le modèle a correctement identifié 98% des prêts qui ont effectivement été remboursés.</li>

                <li><strong>F1-score</strong> : Le F1-score est la moyenne harmonique de la précision et du rappel, offrant un équilibre entre ces deux métriques. 
                Il est particulièrement utile lorsque les classes sont déséquilibrées, car il pénalise à la fois les faux positifs et les faux négatifs. 
                Un F1-score de 91% pour les prêts remboursés indique une forte performance combinée en termes de précision et de rappel.</li>

                <li><strong>AUC-ROC</strong> : L'AUC-ROC (Area Under the Curve - Receiver Operating Characteristic) est une métrique qui évalue la capacité du modèle à distinguer entre les classes. 
                Un score AUC de 0.5 indique une performance aléatoire, tandis qu'un score de 1.0 indique une distinction parfaite. 
                Dans notre cas, un AUC de 0.52 pour le modèle signifie qu'il a du mal à différencier efficacement entre les prêts remboursés et non remboursés.</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)

            elif model_choice == "Decision Tree":
                st.markdown("Le Decision Tree offre des performances similaires, mais tend à surapprendre sur les données d'entraînement.")
            elif model_choice == "Gradient Boosting":
                st.markdown("Le Gradient Boosting améliore légèrement la prédiction des prêts non remboursés grâce à sa capacité d'ensemble.")
            elif model_choice == "Régression Logistique":
                st.markdown("La régression logistique est un modèle simple mais robuste, avec une bonne capacité à prédire les prêts remboursés, bien que la prédiction des non remboursés soit limitée.")
         
        


# Charger les données
data = load_and_encode_data('https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/loan_data_final.csv')

# Préparer les données
X_train_balanced, X_test, y_train_balanced, y_test = prepare_data(data)

# Menu de navigation dans la barre latérale
page = st.sidebar.radio("Navigation", ["Accueil", "Apprentissage supervisé", "Apprentissage non supervisé", "Apprentissage profond"])

if page == "Accueil":
    show_home_page()
elif page == "Apprentissage supervisé":
    show_supervised_learning_page(data, X_train_balanced, y_train_balanced, X_test, y_test)


# Condition pour la page "Apprentissage non supervisé"
elif page == "Apprentissage non supervisé":
    st.title("Apprentissage non supervisé")
    # Ajout du paragraphe Objectif
    st.write("**Objectif :** L'apprentissage non supervisé consiste à entraîner un algorithme d'IA à partir de données non étiquetées, afin de découvrir des structures cachées ou des regroupements dans les données, sans connaître à l'avance notre cible.")
    # Ajout du paragraphe Cas d'usage
    st.write("**Cas d'usage :** Identification des patterns dans l'écosystème de la recherche en Suisse à travers la création de clusters basés sur un jeu de données. (source du jeu données : zenodo.org)")  
    # Chargement du jeu de données spécifique à l'apprentissage non supervisé
    research_final = pd.read_csv('https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/research_final.csv', sep=';')

    # Menu déroulant pour sélectionner la sous-partie
    section_unsupervised = st.selectbox("Sélectionnez une section", 
                                        ["Présentation du jeu de données", "Visualisation", "Modélisation"])
    
    if section_unsupervised == "Présentation du jeu de données":
        st.header("Présentation du jeu de données")

        # Remplacer 'f' par 'female' et 'm' par 'male' dans la colonne 'gender'
        research_final['gender'] = research_final['gender'].replace({'f': 'femme', 'm': 'homme'})

        # Affichage des premières lignes du DataFrame traité
        st.write("Voici les premières lignes du jeu de données après traitement :")
        st.dataframe(research_final.head())
        
        # Affichage des informations sur les variables du dataset
        info = pd.DataFrame({
            "Nom de la variable": research_final.columns,
            "Nombre d'occurrences": research_final.count(),
            "Type de variable": research_final.dtypes
        }).reset_index(drop=True)
        
        info = info.astype(str)
        
        # Centrer le tableau en utilisant Streamlit
        st.markdown("<div style='text-align: center;'>Informations sur les variables du dataset :</div>", unsafe_allow_html=True)
        st.dataframe(info)

        # Ajout du commentaire en dessous du tableau
        st.markdown("### Commentaire")
        st.markdown("""
        <div style="text-align: justify;">
        Le DataFrame, réduit à 14 variables essentielles parmi une trentaine initiale, offre une vue d'ensemble des chercheurs en termes de résultat académique et de production scientifique. 
        Les variables incluent des informations démographiques telles que l'identifiant unique, le genre et le groupe d'âge, ainsi que des détails sur le domaine de recherche principal 
        et le type d'institution d'affiliation. D'autres variables mesurent l'implication dans des projets financés par l'ERC, les publications annuelles, les prépublications, 
        les citations reçues, ainsi que la note moyenne glissante, qui évalue la performance sur une période donnée.
        </div>
        """, unsafe_allow_html=True)

    elif section_unsupervised == "Visualisation":
        st.header("Visualisation")
        # Sélection des colonnes 'max' pour la matrice de corrélation
        max_columns = ['max_rolling_7', 'max_articles_7', 'max_preprints_7', 'max_citations_7']
        correlation_matrix = research_final[max_columns].corr()  # Assurez-vous que 'research_final' est le bon DataFrame
    
        # Affichage de la matrice de corrélation
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matrice de Corrélation des variables articles,citations,prepublications et notes académiques')
        
        # Affichage du graphique dans Streamlit
        st.pyplot(plt.gcf())
        plt.clf()  # Nettoyage du graphique après affichage

        # Ajout du commentaire en dessous de la matrice de corrélation
        st.markdown("### Commentaire matrice de corrélation")
        st.markdown("""
        <div style="text-align: justify;">
        La relation la plus notable est celle entre le nombre maximum d'articles publiés et le nombre maximum de citations, 
        qui montre une corrélation modérée positive. Pour le reste, il y a peu de corrélation entre les variables.
        </div>
        """, unsafe_allow_html=True)

        # Distribution des tranches d'âge par genre
        research_final['gender'] = research_final['gender'].replace({'f': 'femme', 'm': 'homme'})
        age_order = ["< 45", "45-54", "55-64", "65+"]
        gender_order = ["homme", "femme"]

        plt.figure(figsize=(10, 6))
        sns.countplot(data=research_final, x='time_dep_age_group', hue='gender', order=age_order, hue_order=gender_order)
        plt.title('Distribution des Tranches d\'Âge par Genre')
        plt.xlabel('Tranche d\'Âge')
        plt.ylabel('Nombre')

        # Affichage du graphique dans Streamlit
        st.pyplot(plt.gcf())
        plt.clf()

        # Ajout du commentaire en dessous du graphique de distribution des âges
        st.markdown("### Commentaire distribution des âges")
        st.markdown("""
        <div style="text-align: justify;">
        Le jeu de données recense 4460 chercheurs dont 3397 hommes (76%) et 1063 femmes (24%).
        </div>
        """, unsafe_allow_html=True)

        # Graphique de boxplot pour les distributions par genre avec une autre palette
        custom_palette = {"femme": "lightblue", "homme": "lightcoral"}

        # Créer les subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.boxplot(x='gender', y='max_rolling_7', data=research_final, ax=axes[0], hue='gender', palette=custom_palette, dodge=False)
        axes[0].set_title("Distribution de la note glissante maximale par genre")
        axes[0].set_xlabel("Genre")
        axes[0].set_ylabel("Note glissante maximale sur 7 ans")

        sns.boxplot(x='gender', y='max_articles_7', data=research_final, ax=axes[1], hue='gender', palette=custom_palette, dodge=False)
        axes[1].set_title("Distribution du nombre maximum d'articles par genre")
        axes[1].set_xlabel("Genre")
        axes[1].set_ylabel("Nombre maximum d'articles sur 7 ans")

        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

        st.markdown("### Commentaire distribution notes académiques et articles")
        st.markdown("""
        <div style="text-align: justify;">
        <ul>
        <li> La moyenne de la note glissante maximale est légèrement supérieure chez les hommes (3.86) comparée aux femmes (3.65). </li>
        <li> Il existe une différence significative dans le nombre maximum d'articles publiés sur 7 ans, avec une moyenne beaucoup plus élevée chez les hommes (5.21) que chez les femmes (2.72). </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisation du nombre d'articles et de citations par type d'institut et par année
        st.subheader("Nombre d'articles et de citations par type d'institut et par année")
    
        institute_types = research_final['institute_type'].unique()

        fig, ax = plt.subplots(len(institute_types), 2, figsize=(14, len(institute_types) * 5), sharex=True)
    
        for i, institute in enumerate(institute_types):
            institute_data = research_final[research_final['institute_type'] == institute].groupby('year')[['max_articles_7', 'max_citations_7']].sum()

            institute_data['max_articles_7'].plot(ax=ax[i, 0], marker='o', title=f"{institute}: Nombre d'articles publiés par an")
            ax[i, 0].set_ylabel("Nombre d'articles")
            ax[i, 0].grid(True)

            institute_data['max_citations_7'].plot(ax=ax[i, 1], marker='o', title=f"{institute}: Nombre de citations par an")
            ax[i, 1].set_ylabel("Nombre de citations")
            ax[i, 1].grid(True)

            ax[i, 0].set_xticks(institute_data.index)
            ax[i, 1].set_xticks(institute_data.index)
            ax[i, 0].set_xticklabels(institute_data.index.astype(int))
            ax[i, 1].set_xticklabels(institute_data.index.astype(int))
    
        ax[-1, 0].set_xlabel("Année")
        ax[-1, 1].set_xlabel("Année")

        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

        st.markdown("### Commentaire distribution articles vs citations")
        st.markdown("""
        <div style="text-align: justify;">
        <ul>
        <li>Pour les universités cantonales, on observe une croissance constante du nombre d'articles publiés et des citations reçues.</li>
        <li>Le domaine ETH publie moins d'articles, mais reçoit beaucoup de citations, montrant leur influence.</li>
        <li>Les autres instituts ont une productivité scientifique plus modeste.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif section_unsupervised == "Modélisation":
        st.header("Modélisation")

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        if 'research_final' in locals():
            research_final['gender'] = research_final['gender'].replace({'f': 'femme', 'm': 'homme'})
            pca_data = research_final[['anonym_id', 'year', 'gender', 'time_dep_age_group', 'main_research_area', 'institute_type', 'running_erc_project', 'max_rolling_7', 'max_articles_7', 'max_preprints_7', 'max_citations_7']].copy()

            st.write("Les données ont été préparées pour l'analyse de clustering. Voici un aperçu des premières lignes :")
            st.dataframe(pca_data.head())
        else:
            st.error("La variable 'research_final' n'a pas été définie.")

        st.markdown("""
        <p style="text-align: justify;">
        Les données ont été filtrées pour inclure uniquement les colonnes pertinentes au clustering. Ces données serviront de base à l'analyse des similarités entre les chercheurs.
        </p>
        """, unsafe_allow_html=True)

        numeric_data = pca_data[['max_rolling_7', 'max_articles_7', 'max_preprints_7', 'max_citations_7']]
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(numeric_data)
        standardized_data = pd.DataFrame(standardized_data, columns=numeric_data.columns)

        st.write("Données standardisées pour la modélisation :")
        st.dataframe(standardized_data.head())

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(standardized_data)

        variables = standardized_data.columns
        st.subheader("Cercle des corrélations")

        plt.figure(figsize=(8, 8))
        colors = matplotlib.cm.get_cmap('tab10')

        for i, var in enumerate(variables):
            plt.quiver(0, 0, pca.components_[0, i], pca.components_[1, i], angles='xy', scale_units='xy', scale=1, color=colors(i % len(variables)), label=var)

        circle = plt.Circle((0, 0), 1, color='b', fill=False)
        plt.gca().add_artist(circle)

        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Première composante principale (PC1)')
        plt.ylabel('Deuxième composante principale (PC2)')
        plt.title('Cercle des corrélations')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("### Commentaire sur le cercle des corrélations")
        st.markdown("""
        <div style="text-align: justify;">
        <ul>
        <li> La première composante principale (PC1) explique 41.72% de la variance totale.</li>
        <li> Les variables liées aux citations et aux articles sont dominantes dans l'analyse.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Clustering avec K-means")

        from sklearn.cluster import KMeans

        optimal_clusters = 4
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        clusters = kmeans.fit_predict(standardized_data)

        pca_data['Cluster'] = clusters
        cluster_counts = pca_data['Cluster'].value_counts()
        st.write("Répartition des individus par cluster :")
        st.bar_chart(cluster_counts)

        st.subheader("Visualisation des clusters avec PCA")

        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple']

        for cluster in range(optimal_clusters):
            cluster_data = pca_result[pca_data['Cluster'] == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[cluster], label=f'Cluster {cluster}', alpha=0.6, s=100)

        plt.title('Visualisation des clusters après PCA')
        plt.xlabel('Première composante principale (PC1)')
        plt.ylabel('Deuxième composante principale (PC2)')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("### Commentaire sur les clusters")
        st.markdown("""
        <div style="text-align: justify;">
        <li>  Cluster 0 : Productivité modérée mais constante.</li>
        <li> Cluster 1 : Faible productivité.</li>
        <li> Cluster 2 : Similaire au Cluster 0 mais avec moins d'articles publiés.</li>
        <li> Cluster 3 : Le plus productif et influent.</li>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Tests de chi² pour les variables catégorielles vs Cluster")

        import pandas as pd
        from scipy.stats import chi2_contingency

        contingency_table_research_area = pd.crosstab(pca_data['main_research_area'], pca_data['Cluster'])
        chi2_research_area, p_research_area, dof_research_area, _ = chi2_contingency(contingency_table_research_area)

        contingency_table_institute_type = pd.crosstab(pca_data['institute_type'], pca_data['Cluster'])
        chi2_institute_type, p_institute_type, dof_institute_type, _ = chi2_contingency(contingency_table_institute_type)

        contingency_table_age_group = pd.crosstab(pca_data['time_dep_age_group'], pca_data['Cluster'])
        chi2_age_group, p_age_group, dof_age_group, _ = chi2_contingency(contingency_table_age_group)

        st.write("P-values des tests de chi² pour les variables catégorielles :")
        chi2_results = {
            'P-value main_research_area vs Cluster': p_research_area,
            'P-value institute_type vs Cluster': p_institute_type,
            'P-value time_dep_age_group vs Cluster': p_age_group
        }
        st.write(chi2_results)

        st.markdown("### Conclusion sur le cluster")
        st.markdown("""
        <div style="text-align: justify;">
        Les résultats des tests du chi-carré montrent qu’il y a une forte association entre les variables catégorielles (domaine de recherche, type d'institut, et âge) et les clusters.
        </div>
        """, unsafe_allow_html=True)



# Bloc "Apprentissage profond"
elif page == "Apprentissage profond":
    st.title("Big Data et Intelligence Artificielle")

    # Menu déroulant pour les sections d'apprentissage profond
    section_deep_learning = st.selectbox(
    "Sélectionnez une section", 
    ["Contexte et enjeux", "Les Large Language Models", "Cas d'usage"]
    )

    # Section "Contexte et enjeux"
    if section_deep_learning == "Contexte et enjeux":
        st.header("Contexte et Enjeux")
        st.markdown("""
            <div style="text-align: justify;">
            Dans ce portfolio, j'ai exploré deux axes fondamentaux du machine learning : l'apprentissage supervisé 
            et l'apprentissage non supervisé. L'apprentissage par renforcement n'a pas été abordé, en raison des ressources 
            et des volumes de données significativement plus élevés qu'il requiert pour une implémentation robuste.

            Deux modèles ont été développés, chacun basé sur des algorithmes adaptés à des problématiques spécifiques :
        
            - **RandomForestClassifier** : utilisé pour prédire la probabilité de remboursement d'un prêt.
            - **KMeans** : destiné à segmenter les chercheurs en catégories pertinentes dans le cadre d'une étude de positionnement au sein de l'écosystème de la recherche en Suisse.
            </div>
        """, unsafe_allow_html=True)

        # Insertion de l'image à la fin des deux tirets
        st.image("https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/AlgoML.png", caption="Illustration des principaux algorithmes de machine learning", use_column_width=True)
        # Poursuite du texte après l'image
        st.markdown("""
            <div style="text-align: justify;">
            Ce portfolio illustre le processus d'entraînement des modèles, qui, une fois affinés, seront intégrés dans des 
            applications d'aide à la décision. Le modèle de classification permettra de déterminer l'éligibilité des clients 
            à un prêt, tandis que l'algorithme de clustering contribuera à optimiser l'intégration des chercheurs en fonction 
            de leur profil et de leur domaine.
            
            L’actualité récente, avec l’arrivée de ChatGPT, a mis en lumière le deep learning, reposant sur des architectures de réseaux de neurones complexes. Contrairement aux algorithmes classiques comme les Random Forests ou les Decision Trees, souvent interprétables ("white box"), les modèles de deep learning (CNN, RNN, Transformers) sont des "black box" en raison de leur complexité et de leur opacité, ce qui les distingue de l’apprentissage supervisé exploré dans ce portfolio.
            </div>
        """, unsafe_allow_html=True)

    # Section "Les Large Language Models"
    elif section_deep_learning == "Les Large Language Models":
        st.header("Les Large Language Models")
        st.markdown("""
            <div style="text-align: justify;">
            Les algorithmes de deep learning ont connu un essor remarquable avec l'avènement des Large Language Models (LLMs), 
            qui ont révolutionné les techniques de traitement du langage naturel (NLP). Ces modèles ont non seulement établi 
            de nouveaux standards dans diverses tâches liées au langage, mais ils ont également surpassé leurs prédécesseurs 
            en ouvrant la voie à des avancées sans précédent dans l'IA.
            </div>
        """, unsafe_allow_html=True)

        # Insertion de l'image avec une légende
        st.image("https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/IALandscape_2.png", 
            caption="Les LLMs dans le paysage de l'IA", 
            use_column_width=True)

        st.markdown("""
            <div style="text-align: justify;">
            Le qualificatif de "large" ou "grand" pour les LLMs fait référence à l'énorme quantité de données d'entraînement 
            et de ressources nécessaires à leur fonctionnement. Le terme "modèles" souligne leur capacité à apprendre des 
            motifs complexes à partir des données. Dans le cas des LLMs, ces données proviennent principalement de vastes 
            corpus textuels issus d'internet.

            Ces modèles sont également qualifiés de multimodaux car ils sont capables de traiter et de générer des informations 
            à partir de divers types de données, tels que le texte, l'audio, la vidéo, et les images. Cela les différencie des 
            modèles non multimodaux, qui ne fonctionnent qu'avec un seul type de données, généralement du texte.
            </div>
        """, unsafe_allow_html=True)

        # Création des données sous forme de dictionnaire
        data = {
            "Marque": ["Microsoft", "Google", "Google", "Meta (Facebook)", "Meta (Facebook)"],
            "Modèle de Fondation": ["GPT-3", "BERT", "T5", "RoBERTa / BlenderBot", "LLaMA"],
            "Modèle Fine-tuné": [
                "Codex pour la génération de code", 
                "InstructBERT (pour la compréhension des instructions)", 
                "T5 fine-tuné pour des tâches spécifiques comme la traduction ou le résumé de texte", 
                "BlenderBot (tuné pour des conversations humaines)", 
                "LLaMA fine-tuné pour des tâches de génération de texte spécifiques"
            ],
            "Motif": [
                "Génération de code", 
                "Compréhension des instructions", 
                "Traduction ou résumé de texte", 
                "Conversations humaines", 
                "Génération de texte"
            ]
        }

        # Création du DataFrame à partir des données
        df = pd.DataFrame(data)

        # CSS pour centrer et justifier le tableau
        table_style = """
        <style>
        table {
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            text-align: center;
        }
        thead th {
            text-align: center;
        }
        tbody td {
            text-align: justify;
        }
        </style>
        """
        # Affichage du tableau dans Streamlit sans index
        st.markdown("### Modèles de fondation")
        st.markdown(df.to_html(index=False), unsafe_allow_html=True)

        # Texte à afficher entre les deux tableaux avec des tirets sous forme d'astérisques
        texte = """
        La création d’un modèle fondationel (pré-entrainé) nécessite des ordinateurs puissants et une infrastructure spécialisée :
        * Des serveurs puissants : processeurs rapides (CPU), nombreuses unités de traitement graphique (GPU et TPU) 
        * Un stockage massif : systèmes de stockage capables de gérer des téraoctets (1 To équivaut à 1 million de livres de 200 à 500 pages) voire des pétaoctets de données.
        * Des réseaux de haute performance : connexion rapide entre serveurs pour éviter la lenteur lors de la phase d’entraînement 
        * Des logiciels et frameworks adaptés : des environnements cloud (Google Cloud, AWS, Microsoft Azure) et des frameworks (PyTorch, TensorFlow) pour traiter des quantités massives de données nécessitant des ressources computationnelles importantes.
        * La disponibilité des données d'entraînement de haute qualité : garbage in, garbage out
        """

        # Affichage du texte
        st.markdown(texte)

        # Création des données sous forme de dictionnaire
        data = {
            "Critère": ["Puissance", "Temps de préparation", "Quantité de données"],
            "Modèle de fondation": [
                "Des milliers de CPU et GPU", 
                "Des semaines ou des mois", 
                "Des centaines de Go"
             ],
            "Modèle fine-tuné ou spécialisé": [
                "1 CPU ou GPU", 
                "Quelques jours", 
                "Quelques centaines de Mo ou quelques Go"
            ]
        }

        # Création du DataFrame
        df = pd.DataFrame(data)

        # Affichage du tableau dans Streamlit
        st.markdown("### Comparaison entre Modèle de fondation et Modèle fine-tuné ou spécialisé")
        st.markdown(df.to_html(index=False), unsafe_allow_html=True)

        st.image("https://raw.githubusercontent.com/FromTchouaffe/portfolio_new/main/LLMsTrans.png", 
            caption="Schéma de déploiement d'une IA basée sur un LLM", 
            use_column_width=True)

    
        st.markdown("""
            <div style="text-align: justify;">
            Les Large Language Models (LLMs) sont des modèles de traitement du langage naturel (NLP) caractérisés par un très grand nombre de paramètres (allant de plusieurs milliards à des centaines de milliards).
            Cette échelle leur permet de gérer des tâches complexes en capturant des relations fines dans de grandes quantités de données textuelles. Ils sont basés sur l'architecture des Transformers, 
            qui exploitent le mécanisme d'attention pour modéliser efficacement les relations contextuelles entre les mots, même sur de longues séquences textuelles.
            </div>
        """, unsafe_allow_html=True)

    elif section_deep_learning == "Cas d'usage":
        st.header("Chatbot pour la Suite Office")
        st.write("""
            <div style="text-align: justify;">
            Cet assistant est conçu pour aider les utilisateurs des logiciels de la suite Microsoft Office. 
            En exploitant les capacités des modèles de langage, elle permet de répondre en temps réel à des questions techniques spécifiques concernant les outils tels que Word, Excel ou PowerPoint.
            Ce chatbot offre une aide personnalisée et un guidage pas-à-pas pour optimiser l'efficacité des utilisateurs dans leurs tâches quotidiennes.
            </div>
        """, unsafe_allow_html=True)

        import os
        import streamlit as st
        from streamlit_chat import message
        from langchain.chains import ConversationChain
        from langchain_community.llms import OpenAI  # Utilisation de l'import correct


        # Clé API OpenAI (Remplacer par la clé API valide)
        openai_api_key = os.getenv("MY_API_KEY")

        if not openai_api_key:
            st.error("Clé API OpenAI introuvable. Veuillez la définir.")
        else:
        # Charger la chaîne OpenAI
            #lm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = ConversationChain(llm=llm)

        # Initialisation de l'état de session pour stocker les messages
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

        # Créer un formulaire pour la saisie utilisateur
            with st.form(key="user_input_form"):
                user_input = st.text_input("Vous : ", "")
                submit_button = st.form_submit_button(label="Envoyer")

            # Si l'utilisateur envoie une nouvelle question
            if submit_button and user_input:
            # Ajouter le message utilisateur à la session
                st.session_state["messages"].append({"role": "user", "content": user_input})

                try:
                # Générer une réponse avec OpenAI
                    prompt = f"{user_input}\nRépondez toujours en français."
                    output = chain.run(input=prompt)

                # Ajouter la réponse du chatbot à la session
                    st.session_state["messages"].append({"role": "bot", "content": output})
                except Exception as e:
                    st.error(f"Erreur lors de la génération de la réponse : {str(e)}")

            # Afficher les messages dans le chat avec des clés uniques pour chaque message
            for i, msg in enumerate(st.session_state["messages"]):
                if msg["role"] == "user":
                    message(msg["content"], is_user=True, key=f"user_{i}")
                else:
                    message(msg["content"], key=f"bot_{i}")

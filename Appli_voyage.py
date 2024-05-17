import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Charger les données
df_voyage_all = pd.read_parquet(r'C:\Users\aikel\Desktop\Protojam ok\table_voyage_all (1)')
df_name_ml = pd.read_parquet(r'C:\Users\aikel\Desktop\Protojam ok\table_voyage_ml (1)')

# Fonction pour afficher chaque lettre d'une chaîne de caractères avec une couleur différente
def rainbow_text(text):
    colors = ['#FF6633', '#FFB399', '#FF33FF', '#FFFF99', '#00B3E6',
              '#E6B333', '#3366E6', '#999966', '#99FF99', '#B34D4D']
    spans = [f'<span style="color: {colors[i % len(colors)]};">{char}</span>' for i, char in enumerate(text)]
    return ''.join(spans)

# Interface utilisateur avec Streamlit
st.write(f'<h1 style="font-size: 60px;">{rainbow_text("Votre rêve devient réalité")}</h1>', unsafe_allow_html=True)

# Sidebar pour le titre
#st.sidebar.title('Paramètres de voyage')

# Demander à l'utilisateur d'entrer les valeurs
traveler_age = st.sidebar.slider("Âge:", min_value=20, max_value=60, step=1)
traveler_gender = st.sidebar.radio("Genre:", options=["Homme", "Femme"])
duration = st.sidebar.slider("Durée du voyage (jours):", min_value=5, max_value=14, step=1)
accommodation_type_ml = st.sidebar.selectbox("Type d'hébergement:", options=["Hôtel", "Appartement", "Auberge", "Maison", "Camping"])
transportation_type_ml = st.sidebar.selectbox("Moyen de transport:", options=["Avion", "Train", "Voiture", "Bus", "Bateau"])
cost = st.sidebar.number_input("Budget (en $):", min_value=0, step=1)

# Convertir les entrées en valeurs numériques pour la prédiction
gender_mapping = {"Homme": 1, "Femme": 0}
gender_numeric = gender_mapping[traveler_gender]
accommodation_mapping = {"Hôtel": 1, "Appartement": 2, "Auberge": 3, "Maison": 4, "Camping": 5}
accommodation_numeric = accommodation_mapping[accommodation_type_ml]
transportation_mapping = {"Avion": 1, "Train": 2, "Voiture": 3, "Bus": 4, "Bateau": 5}

# Sélectionner les colonnes numériques pour X
X = df_name_ml[["Traveler age", "Traveler gender 0/1", "Duration (days)", "Accommodation type ML", "Transportation type ML", "Total cost"]]

# Construire le modèle ML
model = NearestNeighbors(n_neighbors=3).fit(X)

# Utiliser les valeurs saisies dans la méthode kneighbors()
input_data = [[traveler_age, gender_numeric, duration, accommodation_numeric, transportation_mapping[transportation_type_ml], cost]]
distances, indices = model.kneighbors(input_data)



# Définir le style CSS pour le sous-titre
st.markdown(
    "<h2 style='text-align: center; color: #00008B;'>Pays recommandés pour votre voyage :</h2>", 
    unsafe_allow_html=True
)

# Afficher la destination recommandée (seulement les noms des pays)
nearest_neighbors = df_voyage_all['Country'].iloc[indices[0]].values
st.markdown(f"<h3 style='text-align: center;'>{', '.join(nearest_neighbors)}</h3>", unsafe_allow_html=True)



 # URL de l'image à afficher
image_url = 'https://raw.githubusercontent.com/charaabi01/image-Protajam/main/pp.png'

# Centrer l'image
st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="{image_url}" width="800"></div>',
    unsafe_allow_html=True
)
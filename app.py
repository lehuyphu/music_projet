import streamlit as st
from IdentifiantMusical import generer_melodie_complet
import os

st.set_page_config(page_title="Identité Musicale", page_icon="🎶")

st.title("🎶 Générateur d'Identité Musicale")
st.write("Crée ta mélodie personnalisée à partir de tes informations étudiantes !")

# Formulaire utilisateur
genre = st.selectbox("Genre", ["homme", "femme", "non_binaire"])
filiere = st.selectbox("Filière", [
    "Informatique", 
    "Informatique et Science de l'Ingénieur",
    "Maths", 
    "Maths Info",
    "Physique Chimie",
    "Physique"
])
annee = st.selectbox("Année", ["L1", "L2", "L3", "M1", "M2"])
numero = st.text_input("Numéro étudiant (8 chiffres)")

if st.button("🎼 Générer ma mélodie"):
    if len(numero) != 8 or not numero.isdigit():
        st.error("❌ Le numéro doit contenir exactement 8 chiffres !")
    else:
        notes, durs, bpm = generer_melodie_complet(genre, filiere, annee, numero)

        st.success("✔️ Mélodie générée !")

        # Lecture audio
        if os.path.exists("identite_musicale.wav"):
            audio_file = open("identite_musicale.wav", "rb")
            st.audio(audio_file.read(), format="audio/wav")

        # Téléchargement
        with open("identite_musicale.wav", "rb") as f:
            st.download_button(
                label="⬇️ Télécharger le fichier WAV",
                data=f,
                file_name=f"ID_Musicale_{numero}.wav",
                mime="audio/wav"
            )

        with open("identite_musicale.mid", "rb") as f:
            st.download_button(
                label="⬇️ Télécharger le fichier MIDI",
                data=f,
                file_name=f"ID_Musicale_{numero}.mid",
                mime="audio/midi"
            )

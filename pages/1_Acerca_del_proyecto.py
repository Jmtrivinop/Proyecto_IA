import streamlit as st

st.set_page_config(page_title="Acerca del proyecto", layout="centered")
st.title("ℹ️ Acerca del Proyecto")

st.markdown("""
## Eye-AI con Gemini

Este proyecto utiliza **Gemini 1.5**, el modelo de inteligencia artificial multimodal de Google, 
para realizar análisis avanzados de imágenes, identificar objetos importantes y describir el entorno visual de forma automatizada. 
Además, se integran herramientas como:

- **Google Generative AI (Gemini)** para análisis de imágenes y generación de descripciones.
- **Google Translate** para traducción automática de texto.
- **gTTS** (Google Text-to-Speech) para conversión de texto a voz en español.
- **Streamlit** para una interfaz web interactiva y accesible desde cualquier dispositivo, incluyendo móviles.

### Objetivo

Desarrollar una aplicación de visión computacional basada en inteligencia artificial que analice imágenes del entorno y brinde retroalimentación auditiva,
facilitando la identificación de objetos importantes en tiempo real. Esta solución está pensada especialmente para apoyar a personas con discapacidad visual durante su movilidad diaria.

### Autores
- **Maria Avendaño.**
- **Melissa Muñoz.** 
- **Juan Martin Triviño.**
- **Ricardo Calixto.**
""")

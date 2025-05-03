
import streamlit as st

st.set_page_config(page_title="Acerca del proyecto", layout="centered")
st.title("ℹ️ Acerca del Proyecto")

st.markdown("""
## Eye-AI Vision Computational

Este proyecto usa modelos de **Vision Transformer (ViT)** para realizar tareas de clasificación de imágenes
y detección facial. También integra herramientas como:

- **Mediapipe** para detección facial.
- **Google Translate** para traducción automática de etiquetas.
- **gTTS** para convertir texto a voz.
- **Streamlit** para una interfaz interactiva.

### Objetivo

Desarrollar un sistema automático basado en visión computacional capaz de identificar 
y describir objetos en el entorno cercano de una persona con discapacidad visual que se desplaza 
asistida por un perro guía, proporcionando información en tiempo real mediante retroalimentación auditiva.

Autor: 
- **Maria Avendaño.**
- **Melissa Muñoz.** 
- **Juan Martin Triviño.**
- **Ricardo Calixto.**

""")

# EyeIA - Vision Computacional
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from gtts import gTTS
from deep_translator import GoogleTranslator
import google.generativeai as genai
import base64
import os

# Configurar la página
st.set_page_config(page_title="EyeIA con Gemini", layout="centered")
st.title("👁️ Eye-AI con Gemini")
st.write("Reconocimiento de objetos usando la API de Gemini y traducción a español con voz.")

# Configurar la clave de API de Gemini
# genai.configure(api_key=os.getenv("GEMINI_API_KEY")) Para producción
genai.configure(api_key="AIzaSyCeZvdjXg_-vroduRFkrWt1VWcqlbFUeMc")

def play_audio(text, lang="es"):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    st.audio(audio_bytes.getvalue(), format="audio/mp3")

# Análisis de imagen con Gemini
def analyze_image_with_gemini(image):
    # Convertir la imagen a bytes
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Codificar la imagen en base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # Crear el modelo
    model = genai.GenerativeModel('gemini-1.5-flash')
    # model = genai.GenerativeModel('gemini-1.5-pro')

    # Crear el prompt
    prompt = [
        {
            "mime_type": "image/jpeg",
            "data": encoded_image
        },
        {
            "text": "Describe en detalle los objetos presentes en esta imagen."
        }
    ]

    # Generar la respuesta
    response = model.generate_content(prompt)

    return response.text

# Captura de imagen y procesamiento
col1, col2 = st.columns(2)
with col1:
    imagen = st.camera_input("📸 Captura una imagen")

with col2:
    if imagen:
        st.image(imagen, caption="Imagen capturada", use_container_width=True)
        image = Image.open(imagen)

        st.subheader("📦 Análisis de imagen con Gemini:")
        description = analyze_image_with_gemini(image)
        st.write(description)

        # Traducción y reproducción de voz
        st.subheader("🌎 Traducción y Audio:")
        translated = GoogleTranslator(source="en", target="es").translate(description)
        st.success(f"**{translated}**")
        play_audio(translated)
    else:
        st.info("Por favor, captura una imagen para continuar.")

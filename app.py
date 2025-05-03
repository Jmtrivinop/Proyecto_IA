# EyeIA - Vision Computacional
import streamlit as st
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import numpy as np
from gtts import gTTS
from io import BytesIO
from deep_translator import GoogleTranslator
import cv2
import tempfile
import mediapipe as mp

# Configurar la página
st.set_page_config(page_title="EyeIA", layout="centered")
st.title("👁️ Eye-AI Vision Computational")
st.write("Vision Transformer (ViT) Based - Mejorado con detección facial precisa y traducción.")

# Cargar modelo ViT
MODEL_NAME = "google/vit-base-patch16-224"

@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor

model, processor = load_model()

# Función para reproducir audio en Streamlit de forma segura
def play_audio(text, lang="es"):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    st.audio(audio_bytes.getvalue(), format="audio/mp3")

# Función de detección de rostros usando Mediapipe
def detect_faces(image):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    img_array = np.array(image)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_bgr, detection)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, results.detections is not None and len(results.detections) > 0

# Crear diseño en 2 columnas
col1, col2 = st.columns(2)
with col1:
    imagen = st.camera_input("Captura una imagen")
with col2:
    if imagen is not None:
        st.image(imagen, caption="Imagen capturada", use_container_width=True)

        # Abrir imagen
        image = Image.open(imagen)

        # --- Face Detection ---
        st.subheader("🔍 Detección de rostros:")
        face_img, face_detected = detect_faces(image)

        if face_detected:
            st.image(face_img, caption="Rostros detectados", use_container_width=True)
        else:
            st.warning("No se detectó ninguna cara en la imagen. El flujo continuará.")

        # --- Image Classification ---
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)  # Obtener probabilidades
                predicted_class_idx = probs.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_class_idx]

                # --- Corregir múltiples etiquetas ---
                predicted_label = predicted_label.split(",")[0].split("/")[0].strip()

                confidence = probs[0][predicted_class_idx].item()
            st.subheader("🧠 Predicción:")
            st.success(f"**{predicted_label}** ({confidence * 100:.2f}% de confianza)")

            # --- Traducción ---
            translated_label = GoogleTranslator(source='en', target='es').translate(predicted_label)
            st.subheader("🌎 Traducción al Español:")
            st.info(f"**{translated_label}**")

            # --- Texto a voz ---
            play_audio(translated_label)  # Reproduce el audio automáticamente con la traducción
    else:
        st.info("Por favor, captura una imagen para continuar.")

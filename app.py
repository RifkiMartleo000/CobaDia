import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import random
import os
import logging

# ======== Konfigurasi Logging ========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======== Konfigurasi Halaman ========
st.set_page_config(
    page_title="DRChecker",
    page_icon="🔬",
    layout="wide",
)

# ======== Load Model (sekali saja) ========
@st.cache_resource(allow_failure=True)
def load_model():
    try:
        with open("model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.stop()

# ======== Fungsi Preprocessing Gambar ========
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        st.error(f"Gagal memproses gambar: {str(e)}")
        return None

# ======== Fungsi Penjelasan Hasil ========
def get_explanation(index):
    explanations = [
        "Tidak terdeteksi tanda-tanda Diabetic Retinopathy. Tetap lakukan pemeriksaan rutin setiap tahun.",
        "Terdeteksi DR tingkat ringan. Disarankan kontrol dalam 6-12 bulan.",
        "Terdeteksi DR tingkat sedang. Disarankan kontrol dalam 3-6 bulan.",
        "Terdeteksi DR tingkat parah. Disarankan kontrol dalam 1-3 bulan.",
        "Terdeteksi DR tingkat proliferatif. Segera konsultasikan ke dokter mata."
    ]
    return explanations[index]

# ======== Inisialisasi Session State ========
for key in ["image", "image_bytes", "filename", "name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "name" else ""

# ======== Tema & Font ========
st.sidebar.header("🎨 Kustomisasi Tampilan")
theme_choice = st.sidebar.selectbox("Pilih Mode Tema", ["Default", "Terang", "Gelap"])
font_size = st.sidebar.slider("Ukuran Font (px)", 12, 30, 16)

def set_theme_and_font(theme, font_px):
    if theme == "Terang":
        bg_color = "#ffffff"; text_color = "#000000"
        button_bg_color = "#929292"; button_text_color = "#ffffff"
    elif theme == "Gelap":
        bg_color = "#000000"; text_color = "#ffffff"
        button_bg_color = "#424242"; button_text_color = "#000000"
    else:
        bg_color = "#daffb8"; text_color = "#000000"
        button_bg_color = "#3d8000"; button_text_color = "#ffffff"

    st.markdown(f"""
        <style>
            body, .stApp {{
                background-color: {bg_color};
                color: {text_color};
                font-size: {font_px}px;
            }}
            h1, h2, h3, h4, h5, h6, p, label {{
                color: {text_color};
                font-size: {font_px}px;
            }}
            label {{ font-weight: bold; }}
            div.stButton > button {{
                background-color: {button_bg_color};
                color: {button_text_color};
                font-size: {font_px}px;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                transition: 0.3s;
            }}
            div.stButton > button:hover {{
                background-color: #45a049;
                color: #ffffff;
            }}
        </style>
    """, unsafe_allow_html=True)

    return text_color

text_color = set_theme_and_font(theme_choice, font_size)

# ======== Navigasi ========
st.title("DRChecker 👁")
st.markdown("Website Pendeteksi Diabetic Retinopathy")

option = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Periksa Retina", "Hasil Pemeriksaan", "Tim Kami"]
)

# ======== Halaman Beranda ========
if option == "Beranda":
    st.markdown("<h1>Beranda</h1>", unsafe_allow_html=True)
    st.markdown("<p>Selamat datang di situs Pemeriksaan Diabetic Retinopathy</p>", unsafe_allow_html=True)

    name = st.text_input("Masukkan nama Anda", value=st.session_state["name"])
    if name:
        st.session_state["name"] = name
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Halo, {name}!</p>", unsafe_allow_html=True)

    if st.button("Selesai"):
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Silahkan masuk ke menu Periksa Retina pada bagian 'Pilih Halaman'</p>", unsafe_allow_html=True)

# ======== Halaman Periksa Retina ========
elif option == "Periksa Retina":
    st.markdown("<h1> Periksa Retina </h1>", unsafe_allow_html=True)
    st.markdown("<p> Unggah Gambar Scan Retina Anda </p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            st.session_state["image_bytes"] = uploaded_file.getvalue()
            st.session_state["filename"] = uploaded_file.name
            image = Image.open(io.BytesIO(st.session_state["image_bytes"]))
            st.session_state["image"] = image

            st.success(f"✅ Gambar '{uploaded_file.name}' berhasil diunggah!")
            st.image(image, caption=f"Gambar yang Anda unggah: {uploaded_file.name}", use_container_width=True)
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            st.error(f"Gagal mengunggah gambar: {str(e)}")
    elif st.session_state["image"]:
        st.info(f"Gambar yang telah diunggah sebelumnya: {st.session_state['filename']}")
        st.image(st.session_state["image"], caption=f"Gambar yang telah diunggah: {st.session_state['filename']}", use_container_width=True)
    else:
        st.info("Silakan unggah gambar retina Anda.")

# ======== Halaman Hasil Pemeriksaan ========
elif option == "Hasil Pemeriksaan":
    st.markdown("<h1> Hasil Pemeriksaan </h1>", unsafe_allow_html=True)

    if st.session_state["image"] is None:
        st.warning("Silakan unggah gambar terlebih dahulu di halaman 'Periksa Retina'.")
    else:
        st.image(st.session_state["image"], caption=st.session_state["filename"], use_container_width=True)

        if st.button("🔍 Prediksi"):
            st.info("Memproses gambar...")
            try:
                # Preprocessing & prediksi
                processed_image = preprocess_image(st.session_state["image_bytes"])
                if processed_image is None:
                    st.stop()

                preds = model.predict(processed_image)
                label_idx = np.argmax(preds)
                confidence = float(np.max(preds))

                labels = ["Normal", "Mild", "Moderate", "Severe", "Proliferative DR"]
                st.success(f"Hasil Prediksi: {labels[label_idx]}")
                st.markdown(f"Probabilitas: {confidence:.2%}")
                st.markdown(f"""
                    ### Penjelasan:
                    {get_explanation(label_idx)}
                """)
                logger.info(f"Prediction completed: {labels[label_idx]} with confidence {confidence:.2%}")
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                st.error(f"Gagal melakukan prediksi: {str(e)}")

# ======== Halaman Tim Kami ========
elif option == "Tim Kami":
    st.markdown("<h1> Tim Kami </h1>", unsafe_allow_html=True)
    st.markdown("<h2> El STM </h2>", unsafe_allow_html=True)
    st.markdown("""
        <ul>
            <li>Anggota 1</li>
            <li>Anggota 2</li>
            <li>Anggota 3</li>
        </ul>
    """, unsafe_allow_html=True)

# ======== Footer ========
st.markdown(f"<hr style='border-top: 1px solid {text_color};'>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{text_color};'>drchecker.web@2025</p>", unsafe_allow_html=True)

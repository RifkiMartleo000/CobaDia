import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image as keras_image


# ======== Konfigurasi Halaman ========
st.set_page_config(
    page_title="DRChecker",
    page_icon="🔬",
    layout="wide",
)

# ======== Kustomisasi Tema ========
st.sidebar.header("🎨 Kustomisasi Tampilan")

# Pilihan tema
theme_choice = st.sidebar.selectbox("Pilih Mode Tema", ["Default", "Terang", "Gelap"])

# Ukuran font
font_size = st.sidebar.slider("Ukuran Font (px)", 12, 30, 16)

# CSS injection
def set_theme_and_font(theme, font_px):
    if theme == "Terang":
        bg_color = "#ffffff"
        text_color = "#000000"
        button_bg_color = "#929292"
        button_text_color = "#ffffff"
    elif theme == "Gelap":
        bg_color = "#000000"
        text_color = "#ffffff"
        button_bg_color = "#424242"
        button_text_color = "#000000"
    else:  # Default
        bg_color = "#daffb8"
        text_color = "#000000"
        button_bg_color = "#3d8000"
        button_text_color = "#ffffff"

    st.markdown(f"""
        <style>
            body {{
                background-color: {bg_color};
                color: {text_color};
                font-size: {font_px}px;
            }}
            .stApp {{
                background-color: {bg_color};
                color: {text_color};
                font-size: {font_px}px;
            }}
            h1, h2, h3, h4, h5, h6, p, label {{
                color: {text_color};
                font-size: {font_px}px;
            }}
            label {{
                font-weight: bold;
            }}
            input {{
                background-color: #222;
                color: white;
            }}
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

# ======== Judul dan Navigasi ========
st.title("DRChecker 👁")
st.markdown("Website Pendeteksi Diabetic Retinopathy")

option = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Periksa Retina", "Hasil Pemeriksaan", "Tim Kami"]
)

# ======== Konten Halaman ========
if option == "Beranda":
    st.markdown("<h1>Beranda</h1>", unsafe_allow_html=True)
    st.markdown("<p>Selamat datang di situs Pemeriksaan Diabetic Retinopathy</p>", unsafe_allow_html=True)

    name = st.text_input("Masukkan nama Anda")
    if name:
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Halo, {name}!</p>", unsafe_allow_html=True)

    if st.button("Selesai"):
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Silahkan masuk ke menu Periksa Retina pada bagian 'Pilih Halaman'</p>", unsafe_allow_html=True)

elif option == "Periksa Retina":
    st.markdown("<h1> Periksa Retina </h1>", unsafe_allow_html=True)
    st.markdown("<p> Unggah Gambar Scan Retina Anda </p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state['image'] = image  # Simpan ke session
        st.success("✅ Gambar berhasil diunggah!")
        st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
    else:
        st.info("Silakan unggah gambar dengan format .png, .jpg, atau .jpeg.")

elif option == "Hasil Pemeriksaan":
    st.markdown("<h1> Hasil Pemeriksaan </h1>", unsafe_allow_html=True)

    # Fungsi untuk memuat model
    @st.cache_resource
    def load_model():
        from keras.models import model_from_json
        with open("64x3-CNN.json", "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("64x3-CNN.h5")
        return model

    # Fungsi untuk preprocessing gambar
    def preprocess_image(img, target_size=(64, 64)):
        from keras.preprocessing import image as keras_image
        import numpy as np

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize(target_size)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    # Cek apakah gambar sudah diunggah sebelumnya
    if 'image' not in st.session_state:
        st.warning("Silakan unggah gambar terlebih dahulu di halaman 'Periksa Retina'.")
    else:
        image = st.session_state['image']
        st.image(image, caption="Gambar yang akan diprediksi", use_column_width=True)

        if st.button("🔍 Prediksi"):
            model = load_model()
            processed = preprocess_image(image, target_size=(224, 224))
            prediction = model.predict(processed)
            label_idx = np.argmax(prediction)
            labels = ["Normal", "Mild", "Moderate", "Severe", "Proliferative DR"]

            st.success(f"Hasil Prediksi: **{labels[label_idx]}**")
            st.markdown(f"<p>Probabilitas: <strong>{prediction[0][label_idx]:.2%}</strong></p>", unsafe_allow_html=True)

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

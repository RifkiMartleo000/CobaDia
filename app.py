import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

# ======== Konfigurasi Halaman ========
st.set_page_config(
    page_title="DRChecker",
    page_icon="🔬",
    layout="wide",
)

# Inisialisasi session state jika belum ada
if "image" not in st.session_state:
    st.session_state["image"] = None
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = None
if "filename" not in st.session_state:
    st.session_state["filename"] = None
if "name" not in st.session_state:
    st.session_state["name"] = ""

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

    # Gunakan session_state untuk menyimpan nama
    name = st.text_input("Masukkan nama Anda", value=st.session_state["name"])
    if name:
        st.session_state["name"] = name
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Halo, {name}!</p>", unsafe_allow_html=True)

    if st.button("Selesai"):
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Silahkan masuk ke menu Periksa Retina pada bagian 'Pilih Halaman'</p>", unsafe_allow_html=True)

elif option == "Periksa Retina":
    st.markdown("<h1> Periksa Retina </h1>", unsafe_allow_html=True)
    st.markdown("<p> Unggah Gambar Scan Retina Anda </p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    
    # Jika ada file yang diunggah, simpan ke session_state
    if uploaded_file is not None:
        # Menyimpan bytes data dari gambar
        bytes_data = uploaded_file.getvalue()
        st.session_state["image_bytes"] = bytes_data
        st.session_state["filename"] = uploaded_file.name
        
        # Membuka gambar untuk ditampilkan
        image = Image.open(io.BytesIO(bytes_data))
        st.session_state["image"] = image
        
        st.success(f"✅ Gambar '{uploaded_file.name}' berhasil diunggah!")
        st.image(image, caption=f"Gambar yang Anda unggah: {uploaded_file.name}", use_column_width=True)
    # Jika tidak ada file yang baru diunggah tapi ada di session_state
    elif st.session_state["image"] is not None:
        st.info(f"Gambar yang telah diunggah sebelumnya: {st.session_state['filename']}")
        st.image(st.session_state["image"], caption=f"Gambar yang telah diunggah: {st.session_state['filename']}", use_column_width=True)
    else:
        st.info("Silakan unggah gambar dengan format .png, .jpg, atau .jpeg.")

elif option == "Hasil Pemeriksaan":
    st.markdown("<h1> Hasil Pemeriksaan </h1>", unsafe_allow_html=True)

    if st.session_state["image"] is None:
        st.warning("Silakan unggah gambar terlebih dahulu di halaman 'Periksa Retina'.")
    else:
        st.image(st.session_state["image"], caption=f"Gambar yang akan diprediksi: {st.session_state['filename']}", use_column_width=True)

        if st.button("🔍 Prediksi"):
            # Placeholder untuk fungsi preprocessing dan model prediksi
            # Di sini kita perlu menambahkan fungsi preprocess_image dan model yang diimpor
            st.info("Sedang memproses gambar...")
            
            # Simulasi hasil prediksi (ganti dengan model sebenarnya)
            import time
            time.sleep(1)  # Simulasi waktu pemrosesan
            
            # Contoh hasil prediksi (ganti dengan model prediksi sebenarnya)
            import random
            labels = ["Normal", "Mild", "Moderate", "Severe", "Proliferative DR"]
            label_idx = random.randint(0, 4)
            prediction_score = random.uniform(0.7, 0.98)
            
            st.success(f"Hasil Prediksi: {labels[label_idx]}")
            st.markdown(f"Probabilitas: {prediction_score:.2%}")
            
            # Tambahkan penjelasan tentang hasil (opsional)
            st.markdown(f"""
            ### Informasi tentang {labels[label_idx]} Diabetic Retinopathy:
            
            {get_explanation(label_idx)}
            """)


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

# Fungsi untuk mendapatkan penjelasan hasil
def get_explanation(index):
    explanations = [
        "Tidak terdeteksi tanda-tanda Diabetic Retinopathy. Tetap lakukan pemeriksaan rutin setiap tahun.",
        "Terdeteksi Diabetic Retinopathy tingkat ringan. Disarankan untuk kontrol dalam 6-12 bulan.",
        "Terdeteksi Diabetic Retinopathy tingkat sedang. Disarankan untuk kontrol dalam 3-6 bulan.",
        "Terdeteksi Diabetic Retinopathy tingkat parah. Disarankan untuk kontrol dalam 1-3 bulan.",
        "Terdeteksi Diabetic Retinopathy tingkat proliferatif. Perlu segera konsultasi dengan dokter spesialis mata."
    ]
    return explanations[index]

# ======== Footer ========
st.markdown(f"<hr style='border-top: 1px solid {text_color};'>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{text_color};'>drchecker.web@2025</p>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

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
    elif theme == "Gelap":
        bg_color = "#0e1117"
        text_color = "#ffffff"
    else:  # Default
        bg_color = "#f0f2f6"
        text_color = "#31333F"
    
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
            h1, h2, h3, h4, h5, h6 {{
                color: {text_color};
            }}
        </style>
    """, unsafe_allow_html=True)

set_theme_and_font(theme_choice, font_size)

# ======== Judul dan Navigasi ========
st.title("DRChecker 👁")
st.markdown("Website Pendeteksi Diabetic Retinopathy")

option = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Periksa Retina", "Hasil Pemeriksaan", "Tim Kami"]
)

# ======== Konten Halaman ========
if option == "Beranda":
    st.header("Beranda")
    st.write("Selamat datang di situs Pemeriksaan Diabetic Retinopathy")
    st.write("Gunakan sidebar untuk navigasi ke halaman lain.")
    
    name = st.text_input("Masukkan nama Anda")
    if name:
        st.write(f"Halo, {name}!")
    
    if st.button("Klik Saya"):
        st.balloons()
        st.write("Terima kasih telah mengklik tombol!")

elif option == "Periksa Retina":
    st.header("Periksa Retina")
    st.subheader("Unggah Gambar Retina Anda")
    
    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.success("✅ Gambar berhasil diunggah!")
        st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
    else:
        st.info("Silakan unggah gambar dengan format .png, .jpg, atau .jpeg.")

elif option == "Hasil Pemeriksaan":
    st.header("Hasil Pemeriksaan")
    
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    viz_type = st.radio(
        "Pilih jenis visualisasi",
        ["Line Chart", "Bar Chart", "Area Chart"]
    )
    
    if viz_type == "Line Chart":
        st.line_chart(chart_data)
    elif viz_type == "Bar Chart":
        st.bar_chart(chart_data)
    else:
        st.area_chart(chart_data)
    
    st.subheader("Visualisasi Custom dengan Matplotlib")
    fig, ax = plt.subplots()
    ax.scatter(chart_data.index, chart_data['A'], color='red', label='A')
    ax.scatter(chart_data.index, chart_data['B'], color='blue', label='B')
    ax.legend()
    ax.set_title("Scatter Plot")
    ax.set_xlabel("Index")
    ax.set_ylabel("Nilai")
    st.pyplot(fig)

else:
    st.header("Tim Kami")
    st.write("""
    ### El STM

    - Anggota 1
    - Anggota 2
    - Anggota 3
    """)

# ======== Footer ========
st.markdown("---")
st.markdown("drchecker.web@2025")

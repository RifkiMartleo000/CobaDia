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
        button_bg_color = "#929292"
        button_text_color = "#ffffff"

    elif theme == "Gelap":
        bg_color = "#000000"
        text_color = "#ffffff"
        button_bg_color = "#424242"
        button_text_color = "#000000"

    else:  # Default
        bg_color = "#5dc200"
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
            h1 {{
                color: {text_color};
                font-size: 40px;
            }}
            h2 {{
                color: {text_color};
                font-size: 35px;
            }}
            p {{
                color: {text_color};
                font-size: {font_px}px;
            }}
            label {{
                color: white !important;  
                font-weight: bold;
                font-size: 16px;
            }}
            input {{
                background-color: #222;   
                color: white;             
            }}
            div.stButton > button {{
                background-color: {button_bg_color};
                color: {button_text_color};
                font-size: {font_size}px;
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
    st.markdown("<h1>Beranda</h1>", unsafe_allow_html=True)
    st.markdown("<p>Selamat datang di situs Pemeriksaan Diabetic Retinopathy</p>", unsafe_allow_html=True)

    name = st.text_input("Masukkan nama Anda")
    if name:
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Halo, {name}!")
        
    if st.button("Selesai"):
        st.markdown(f"<p style='color:{text_color}; font-size:{font_size}px;'>Silahkan masuk ke menu Periksa Retina pada bagian 'Pilih Halaman'</p>", unsafe_allow_html=True)


elif option == "Periksa Retina":
    st.markdown("<h1> Periksa Retina </h1>", unsafe_allow_html=True)
    st.markdown("<p> Unggah Gambar Retina Anda </p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.success("✅ Gambar berhasil diunggah!")
        st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
    else:
        st.info("Silakan unggah gambar dengan format .png, .jpg, atau .jpeg.")

elif option == "Hasil Pemeriksaan":
    st.markdown("<h1> Hasil Pemeriksaan </h1>", unsafe_allow_html=True)
    
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
    st.markdown("<h1> Tim Kami </h1>", unsafe_allow_html=True)
    st.markdown("<h2> El STM </h2>", unsafe_allow_html=True)

    st.markdown("""
    ### Tim Kami

    - Anggota 1  
    - Anggota 2  
    - Anggota 3
    """)

# ======== Footer ========
st.markdown("---")
st.markdown("drchecker.web@2025")

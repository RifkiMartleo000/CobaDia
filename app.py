import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Set page title and configuration
st.set_page_config(
    page_title="DRChecker",
    page_icon="📊",
    layout="wide",
)

# Title and description
st.title("DRChecker")
st.markdown("Website Pendeteksi Diabetic Retinopathy")

# Sidebar
st.sidebar.header("Pengaturan")
option = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Periksa Retina", "Hasil Pemeriksaan", "Tim Kami"]
)

# Main content based on selected page
if option == "Beranda":
    st.header("Beranda")
    st.write("Selamat datang di aplikasi web Streamlit sederhana!")
    st.write("Gunakan sidebar untuk navigasi ke halaman lain.")
    
    # Interactive elements
    name = st.text_input("Masukkan nama Anda")
    if name:
        st.write(f"Halo, {name}!")
    
    # Button example
    if st.button("Klik Saya"):
        st.balloons()
        st.write("Terima kasih telah mengklik tombol!")
    
elif option == "Periksa Retina":
    st.header("Periksa Retina")
    
    st.subheader("Contoh Data")
    
    uploaded_file = st.file_uploader("Pilih gambar untuk diunggah", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.success("✅ Gambar berhasil diunggah!")
        st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
    else:
        st.info("Silakan unggah gambar dengan format .png, .jpg, atau .jpeg.")

elif option == "Hasil Pemeriksaan":
    st.header("Hasil Pemeriksaan")
    
    # Generate random data for visualization
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    # Show different visualization options
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
    
    # Custom matplotlib figure
    st.subheader("Visualisasi Custom dengan Matplotlib")
    fig, ax = plt.subplots()
    ax.scatter(chart_data.index, chart_data['A'], color='red', label='A')
    ax.scatter(chart_data.index, chart_data['B'], color='blue', label='B')
    ax.legend()
    ax.set_title("Scatter Plot")
    ax.set_xlabel("Index")
    ax.set_ylabel("Nilai")
    st.pyplot(fig)

else:  # About page
    st.header("Tim Kami")
    st.write("""
    ### El STM
    
    - Fayzul Haq Mahardika Basunjaya
    - Kevin Surya Prayoga Wibowo
    - Rifki Martleo Alfiansyah
    
    """)
    
   
# Footer
st.markdown("---")
st.markdown("drchecker.web@2025")

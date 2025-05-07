import streamlit as st
import pandas as pd
import numpy as np

# Set page title and configuration
st.set_page_config(
    page_title="Aplikasi Web Streamlit Sederhana",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Aplikasi Web Streamlit Sederhana")
st.markdown("Aplikasi web sederhana menggunakan Streamlit untuk visualisasi data")

# Sidebar
st.sidebar.header("Pengaturan")
option = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Beranda", "Data", "Visualisasi", "Tentang"]
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

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Anda unggah", use_column_width=True)
    else:
        st.info("Silakan unggah file gambar (.png, .jpg, atau .jpeg)")

elif option == "Data":
    st.header("Data")
    
    # Generate sample data
    st.subheader("Contoh Data")
    data_size = st.slider("Jumlah data", 5, 100, 20)
    
    # Create sample dataframe
    data = pd.DataFrame({
        'Tanggal': pd.date_range(start='2023-01-01', periods=data_size),
        'Nilai': np.random.randn(data_size).cumsum(),
        'Kategori': np.random.choice(['A', 'B', 'C'], size=data_size)
    })
    
    st.dataframe(data)
    
    # Download button
    csv = data.to_csv(index=False)
    st.download_button(
        label="Unduh data sebagai CSV",
        data=csv,
        file_name="data_sample.csv",
        mime="text/csv"
    )

elif option == "Visualisasi":
    st.header("Visualisasi")
    
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
    st.header("Tentang")
    st.write("""
    ### Aplikasi Web Streamlit
    
    Ini adalah aplikasi web sederhana yang dibuat menggunakan Streamlit.
    
    **Fitur:**
    - Visualisasi data interaktif
    - Analisis data sederhana
    - Antarmuka pengguna yang ramah
    
    Untuk informasi lebih lanjut, kunjungi [Streamlit](https://streamlit.io).
    """)
    
    # Contact form
    st.subheader("Hubungi Kami")
    contact_form = st.form("contact_form")
    name = contact_form.text_input("Nama")
    email = contact_form.text_input("Email")
    message = contact_form.text_area("Pesan")
    submit = contact_form.form_submit_button("Kirim")
    
    if submit:
        st.success("Pesan Anda telah dikirim! (Ini hanya simulasi, tidak ada pesan yang benar-benar dikirim)")

# Footer
st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit")

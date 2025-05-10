import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import cv2
import os
import requests
import tempfile

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
if "model" not in st.session_state:
    # Struktur model dalam format JSON sebagai fallback
    model_json = """{"module": "keras", "class_name": "Sequential", "config": {"name": "sequential", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "layers": [{"module": "keras.layers", "class_name": "InputLayer", "config": {"batch_shape": [null, 224, 224, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "registered_name": null}, {"module": "keras.layers", "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 224, 224, 3]}}, {"module": "keras.layers", "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "registered_name": null}, {"module": "keras.layers", "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "moving_mean_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "moving_variance_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null, "synchronized": false}, "registered_name": null, "build_config": {"input_shape": [null, 111, 111, 8]}}, {"module": "keras.layers", "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 111, 111, 8]}}, {"module": "keras.layers", "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "registered_name": null}, {"module": "keras.layers", "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "moving_mean_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "moving_variance_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null, "synchronized": false}, "registered_name": null, "build_config": {"input_shape": [null, 54, 54, 16]}}, {"module": "keras.layers", "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 32, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 54, 54, 16]}}, {"module": "keras.layers", "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "registered_name": null}, {"module": "keras.layers", "class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": -1, "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "moving_mean_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "moving_variance_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null, "synchronized": false}, "registered_name": null, "build_config": {"input_shape": [null, 25, 25, 32]}}, {"module": "keras.layers", "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "data_format": "channels_last"}, "registered_name": null, "build_config": {"input_shape": [null, 25, 25, 32]}}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 20000]}}, {"module": "keras.layers", "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "rate": 0.15, "seed": null, "noise_shape": null}, "registered_name": null}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 32]}}], "build_input_shape": [null, 224, 224, 3]}, "registered_name": null, "build_config": {"input_shape": [null, 224, 224, 3]}, "compile_config": {"optimizer": {"module": "keras.optimizers", "class_name": "Adam", "config": {"name": "adam", "learning_rate": 9.999999747378752e-06, "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "loss_scale_factor": null, "gradient_accumulation_steps": null, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}, "registered_name": null}, "loss": {"module": "keras.losses", "class_name": "BinaryCrossentropy", "config": {"name": "binary_crossentropy", "reduction": "sum_over_batch_size", "from_logits": false, "label_smoothing": 0.0, "axis": -1}, "registered_name": null}, "loss_weights": null, "metrics": ["acc"], "weighted_metrics": null, "run_eagerly": false, "steps_per_execution": 1, "jit_compile": false}}"""
    
    # Path ke file model .h5
    model_path = "model_dr_classifier.h5"
    
    with st.spinner("Memuat model..."):
        try:
            # Coba load model dari file .h5 jika tersedia
            if os.path.exists(model_path):
                # Load model langsung dari file .h5 (struktur + bobot)
                model = keras.models.load_model(model_path)
                # Kompilasi model
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                # Simpan model di session state
                st.session_state["model"] = model
                st.session_state["model_loaded"] = True
                st.success(f"Model berhasil dimuat dari {model_path}")
            else:
                # Coba cari di folder "models" jika ada
                alt_model_path = os.path.join("models", "model_dr_classifier.h5")
                if os.path.exists(alt_model_path):
                    model = keras.models.load_model(alt_model_path)
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    st.session_state["model"] = model
                    st.session_state["model_loaded"] = True
                    st.success(f"Model berhasil dimuat dari {alt_model_path}")
                else:
                    # Jika tidak ada file model, gunakan struktur JSON sebagai fallback
                    st.warning("File model .h5 tidak ditemukan. Menggunakan model dari JSON (tanpa bobot yang dilatih).")
                    model = model_from_json(model_json)
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    st.session_state["model"] = model
                    st.session_state["model_loaded"] = True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.session_state["model_loaded"] = False

# ======== Sidebar untuk Admin ========
if st.sidebar.checkbox("Admin Panel", False):
    st.sidebar.subheader("Administrator Panel")
    
    # Upload model baru
    uploaded_model = st.sidebar.file_uploader("Upload model (.h5)", type=['h5'])
    
    if uploaded_model is not None:
        with st.sidebar.spinner("Memproses model..."):
            try:
                # Buat folder models jika belum ada
                if not os.path.exists("models"):
                    os.makedirs("models")
                
                # Simpan model yang diunggah
                model_save_path = os.path.join("models", "model_dr_classifier.h5")
                with open(model_save_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                # Muat model dari file yang disimpan
                model = keras.models.load_model(model_save_path)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                st.session_state["model"] = model
                st.session_state["model_loaded"] = True
                st.sidebar.success(f"Model berhasil disimpan ke {model_save_path} dan dimuat!")
            except Exception as e:
                st.sidebar.error(f"Error saat menyimpan atau memuat model: {e}")

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
        st.image(image, caption=f"Gambar yang Anda unggah: {uploaded_file.name}", use_container_width=True)
    # Jika tidak ada file yang baru diunggah tapi ada di session_state
    elif st.session_state["image"] is not None:
        st.info(f"Gambar yang telah diunggah sebelumnya: {st.session_state['filename']}")
        st.image(st.session_state["image"], caption=f"Gambar yang telah diunggah: {st.session_state['filename']}", use_container_width=True)
    else:
        st.info("Silakan unggah gambar dengan format .png, .jpg, atau .jpeg.")

elif option == "Hasil Pemeriksaan":
    st.markdown("<h1> Hasil Pemeriksaan </h1>", unsafe_allow_html=True)

    if st.session_state["image"] is None:
        st.warning("Silakan unggah gambar terlebih dahulu di halaman 'Periksa Retina'.")
    else:
        st.image(st.session_state["image"], caption=f"Gambar yang akan diprediksi: {st.session_state['filename']}", use_container_width=True)

        # Tombol untuk memprediksi
        if st.button("🔍 Prediksi"):
            if "model_loaded" in st.session_state and st.session_state["model_loaded"]:
                # Tampilkan spinner selama pemrosesan
                with st.spinner("Memproses gambar..."):
                    try:
                        # Preprocessing gambar
                        processed_img = preprocess_image(st.session_state["image"])
                        
                        # Prediksi
                        prediction = st.session_state["model"].predict(processed_img, verbose=0)
                        
                        # Hasil prediksi (biner)
                        class_idx = np.argmax(prediction[0])
                        class_prob = prediction[0][class_idx]
                        
                        # Tentukan label
                        labels = ["Normal", "Diabetic Retinopathy"]
                        result_label = labels[class_idx]
                        
                        # Tampilkan hasil
                        st.success(f"Hasil Prediksi: {result_label}")
                        st.markdown(f"Probabilitas: {class_prob:.2%}")
                        
                        # Tampilkan penjelasan
                        st.markdown(f"""
                        ### Informasi tentang hasil:
                        
                        {get_explanation(prediction[0])}
                        """)
                        
                        # Menampilkan grafik probabilitas
                        chart_data = pd.DataFrame({
                            'Kelas': labels,
                            'Probabilitas': [prediction[0][0], prediction[0][1]]
                        })
                        
                        st.bar_chart(chart_data.set_index('Kelas'))
                        
                    except Exception as e:
                        st.error(f"Error dalam pemrosesan gambar: {e}")
            else:
                st.error("Model tidak berhasil dimuat. Coba muat ulang aplikasi.")
                st.info("Pastikan Anda telah menginstal TensorFlow dan Keras.")



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

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    # Konversi PIL Image ke array numpy
    img_array = np.array(image)
    
    # Pastikan gambar adalah RGB (3 channel)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        # Konversi grayscale ke RGB jika perlu
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        # Atau hapus channel alpha jika ada 4 channel
        elif img_array.shape[2] == 4:
            img_array = img_array[:,:,:3]
    
    # Resize ke ukuran 224x224 (input model)
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalisasi nilai piksel ke range [0, 1]
    img_normalized = img_resized / 255.0
    
    # Expand dimensi untuk batch (1, 224, 224, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Fungsi untuk mendapatkan penjelasan hasil
def get_explanation(prediction):
    # Karena model menghasilkan output 2 kelas
    if prediction[1] > 0.5:  # Class 1 - Diabetic Retinopathy
        return "Terdeteksi Diabetic Retinopathy. Gambar menunjukkan tanda-tanda kelainan pada retina. Sebaiknya konsultasikan dengan dokter mata untuk evaluasi lebih lanjut."
    else:  # Class 0 - Normal
        return "Tidak terdeteksi Diabetic Retinopathy. Retina tampak normal. Tetap lakukan pemeriksaan rutin setiap tahun."

# ======== Footer ========
st.markdown(f"<hr style='border-top: 1px solid {text_color};'>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{text_color};'>drchecker.web@2025</p>", unsafe_allow_html=True)

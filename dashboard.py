import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Pastikan nama file model tanpa spasi ya!
    yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Balqis Isaura_Laporan2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka dan tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE 1: Deteksi Objek (YOLO)
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    # ==========================
    # MODE 2: Klasifikasi Gambar (TensorFlow)
    # ==========================
    elif menu == "Klasifikasi Gambar":
        # --- Preprocessing ---
        img = img.convert("RGB")                       # pastikan format RGB
        img_resized = img.resize((224, 224))           # sesuaikan dengan input model
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # tambahkan batch dimension

        # --- Antisipasi bug dtype tuple ---
        if isinstance(img_array, tuple):
            img_array = np.array(img_array[0], dtype=np.float32)

        # --- Prediksi ---
        prediction = classifier.predict(img_array)
        class_index = int(np.argmax(prediction))
        probability = float(np.max(prediction))

        # --- Output ke UI ---
        st.write("### 🧩 Hasil Prediksi:", class_index)
        st.write("Probabilitas:", f"{probability:.4f}")

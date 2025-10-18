import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        # Load YOLO model
        yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")

        # Load Keras classifier safely
        classifier = tf.keras.models.load_model(
            "model/Balqis Isaura_Laporan2.h5",
            compile=False,
            safe_mode=True  # mode aman untuk model .h5 lama
        )

        return yolo_model, classifier

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# Panggil fungsi load
yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        img_array = np.array(img)
        results = yolo_model(img_array)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        try:
            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.success(f"Hasil Prediksi: {class_index}")
            st.write(f"Probabilitas: {confidence:.4f}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

else:
    st.info("Silakan unggah gambar terlebih dahulu.")

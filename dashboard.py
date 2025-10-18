# app.py
import streamlit as st
import torch
import tensorflow as tf

st.title("Dashboard Model - Balqis Isaura")

# Sidebar untuk pilih model
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["PyTorch (.pt)", "TensorFlow (.h5)"]
)

# Load Model berdasarkan pilihan
if model_choice == "PyTorch (.pt)":
    st.header("Model PyTorch")
    try:
        model = torch.load('Balqis Isaura_Laporan 4.pt', map_location='cpu')
        st.success("✅ Model PyTorch berhasil dimuat!")
        st.write(f"Tipe model: {type(model)}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

elif model_choice == "TensorFlow (.h5)":
    st.header("Model TensorFlow")
    try:
        model = tf.keras.models.load_model('Balqis Isaura_Laporan2.h5')
        st.success("✅ Model TensorFlow berhasil dimuat!")
        
        # Tampilkan summary model
        from io import StringIO
        stream = StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        st.text(stream.getvalue())
    except Exception as e:
        st.error(f"❌ Error: {e}")

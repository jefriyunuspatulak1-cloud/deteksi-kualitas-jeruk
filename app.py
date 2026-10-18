
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Deteksi Kualitas Jeruk",
    page_icon="🍊"
)

MODEL_PATH = "model_jeruk_mobilenetv2.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🍊 Deteksi Kualitas dan Kerusakan Jeruk")
st.write("Unggah foto jeruk untuk melihat hasil klasifikasi.")

uploaded_file = st.file_uploader(
    "Pilih foto jeruk",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Foto jeruk", use_container_width=True)

    if st.button("Prediksi Jeruk"):
        model = load_model()

        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)[0]
        class_names = [
            "blackspotted_orange",
            "cankered_orange",
            "fresh_orange",
            "green_orange",
            "rotten_orange"
        ]

        index = int(np.argmax(prediction))
        confidence = float(prediction[index]) * 100

        st.subheader("Hasil Prediksi")
        st.write("Kategori:", class_names[index])
        st.metric("Skor keyakinan", f"{confidence:.2f}%")
        st.caption(
            "Skor keyakinan bukan jaminan bahwa prediksi selalu benar."
        )

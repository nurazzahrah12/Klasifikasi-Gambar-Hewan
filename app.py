import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Judul aplikasi
st.title("Klasifikasi Hewan - Animals10 (MobileNetV2)")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animals10_mobilenetv2.h5")
    return model

model = load_model()

# Kelas hewan (urut sesuai training)
class_names = ['Anjing', 'Ayam', 'Domba', 'Gajah', 'Kucing',
               'Kuda', 'Kupu-kupu', 'Laba-laba', 'Sapi', 'Tupai']

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar hewan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    # Hasil
    st.markdown(f"### Prediksi: `{predicted_class}`")
    st.markdown(f"**Confidence**: `{confidence:.2f}%`")
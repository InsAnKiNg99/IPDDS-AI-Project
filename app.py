import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("Intelligent Plant Disease Detection")
st.write("Upload a clear picture of a plant leaf, and our AI will determine its health status.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/disease_model.h5")
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    st.write("Analyzing...")

    image_resized = image.resize((256, 256))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction_scores = model.predict(image_array)
    
    class_names = ["Healthy", "Early Blight", "Late Blight"]
    
    predicted_index = np.argmax(prediction_scores)
    final_result = class_names[predicted_index]
    
    st.success(f"Prediction: **{final_result}**")
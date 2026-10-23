import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page configuration and header
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("Intelligent Plant Disease Detection")
st.write("Upload a clear picture of a plant leaf, and our AI will determine its health status.")

# 2. Load the trained AI model
# The @st.cache_resource decorator prevents the model from reloading every time a user clicks a button
@st.cache_resource
def load_model():
    # Ensure this path matches where you saved your downloaded Colab model
    model = tf.keras.models.load_model("model/disease_model.h5")
    return model

model = load_model()

# 3. Create the file uploader interface
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 4. Display the uploaded image on the screen
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    st.write("Analyzing...")

    # 5. Preprocess the image to match how the AI was trained
    # Note: Change (256, 256) if you used a different image size in Google Colab
    image_resized = image.resize((256, 256))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0) # Adds a batch dimension
    
    # 6. Make the prediction
    prediction_scores = model.predict(image_array)
    
    # Example categories (You must replace these with the actual categories your model learned)
    class_names = ["Healthy", "Early Blight", "Late Blight"]
    
    # Identify the highest probability score
    predicted_index = np.argmax(prediction_scores)
    final_result = class_names[predicted_index]
    
    # 7. Display the result to the user
    st.success(f"Prediction: **{final_result}**")
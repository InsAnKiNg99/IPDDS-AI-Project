import os
import json
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="IPDDS | Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #40916c;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .status-healthy {
        background-color: #d8f3dc;
        color: #1b4332;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .status-diseased {
        background-color: #ffe5d9;
        color: #9d0208;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .card-box {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #2d6a4f;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Supported Potato & Tomato Classes Fallback
DEFAULT_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

DEFAULT_METADATA = {
    "Potato___Early_blight": {
        "display_name": "Potato: Early Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria solani)",
        "recommendation": "Remove lower infected leaves, ensure adequate crop rotation, and apply copper-based fungicides."
    },
    "Potato___Late_blight": {
        "display_name": "Potato: Late Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Water mold (Phytophthora infestans)",
        "recommendation": "High severity. Destroy infected plants immediately, improve soil drainage, and avoid wet foliage."
    },
    "Potato___healthy": {
        "display_name": "Potato: Healthy",
        "crop": "Potato",
        "status": "Healthy",
        "pathogen": "None",
        "recommendation": "Leaves appear completely healthy. Continue standard watering and fertilizing schedules."
    },
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato: Bacterial Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Bacteria (Xanthomonas spp.)",
        "recommendation": "Avoid overhead watering, sanitize tools between pruning, and apply fixed copper sprays."
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato: Early Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria linariae)",
        "recommendation": "Mulch the soil surface, prune bottom leaves to stop splash-back, and use organic neem or fungicides."
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato: Late Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "recommendation": "Act quickly: prune damaged vines, keep canopy aerated, and apply preventative bio-fungicides."
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato: Leaf Mold",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Passalora fulva)",
        "recommendation": "Improve ventilation, lower relative humidity below 85%, and use resistant tomato varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato: Septoria Leaf Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Septoria lycopersici)",
        "recommendation": "Remove infected lower foliage, avoid wetting foliage during irrigation, and rotate crops."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato: Spider Mites",
        "crop": "Tomato",
        "status": "Infested",
        "pathogen": "Pest (Tetranychus urticae)",
        "recommendation": "Apply insecticidal soap or horticultural oils under leaves. Release predatory mites if indoors."
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato: Target Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Corynespora cassiicola)",
        "recommendation": "Improve airflow between plants, keep foliage dry, and apply targeted copper/mancozeb fungicides."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato: Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Virus (TYLCV transmitted by Whiteflies)",
        "recommendation": "Use yellow sticky traps and insect netting against whiteflies. Remove and burn heavily infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato: Mosaic Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Tobamovirus (Mechanically transmitted)",
        "recommendation": "Disinfect tools with 10% bleach, avoid handling plants when damp, and remove infected plants."
    },
    "Tomato___healthy": {
        "display_name": "Tomato: Healthy",
        "crop": "Tomato",
        "status": "Healthy",
        "pathogen": "None",
        "recommendation": "Leaves appear vibrant and healthy. Maintain consistent moisture and balanced nitrogen levels."
    }
}

# 3. Model & Metadata Loading
@st.cache_resource
def load_trained_model():
    model_paths = [
        os.path.join("model", "disease_model.keras"),
        os.path.join("model", "disease_model.h5")
    ]
    for p in model_paths:
        if os.path.exists(p):
            try:
                # Check file size to avoid loading dummy text files
                if os.path.getsize(p) > 1024 * 100:
                    model = tf.keras.models.load_model(p)
                    return model, p
            except Exception as e:
                st.warning(f"Failed to load {p}: {e}")
    return None, None

def load_metadata():
    json_path = os.path.join("model", "class_indices.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("class_names", DEFAULT_CLASSES), data.get("classes", DEFAULT_METADATA)
        except Exception:
            pass
    return DEFAULT_CLASSES, DEFAULT_METADATA

class_names, metadata_map = load_metadata()
model, loaded_path = load_trained_model()

# 4. Sidebar Controls & Info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=400&auto=format&fit=crop&q=80", width="stretch")
    st.title("🌿 IPDDS Overview")
    st.write("**Intelligent Plant Disease Detection System** specialized for Potato & Tomato crops.")
    
    st.divider()
    st.write("### Model Status")
    if model is not None:
        st.success(f"Loaded: `{os.path.basename(loaded_path)}`")
    else:
        st.warning("No trained model weights found yet.")
        st.info("Run `python train.py` to train the model on the Potato & Tomato dataset.")

    st.divider()
    st.write(f"### Supported Classes ({len(class_names)})")
    st.markdown("- **Potato**: Early Blight, Late Blight, Healthy")
    st.markdown("- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Mosaic Virus, TYLCV, Healthy")

# 5. Main UI
st.markdown('<div class="main-header">🌿 Intelligent Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a photo of a Tomato or Potato leaf for real-time AI disease diagnosis and agronomic recommendations.</div>', unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("1. Leaf Image Input")
    uploaded_file = st.file_uploader(
        "Choose a clear leaf photo (JPG, PNG, JPEG)...",
        type=["jpg", "jpeg", "png"],
        help="Make sure the leaf is well-lit and centered in the image frame."
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Preview", width="stretch")

with col_result:
    st.subheader("2. Diagnostic Analysis")
    
    if uploaded_file is None:
        st.info("👈 Upload an image on the left to begin diagnosis.")
    elif model is None:
        st.warning("⚠️ The AI model is not trained yet!")
        st.markdown("""
        To train the model on your downloaded **PlantVillage Potato & Tomato** dataset:
        1. Open your terminal in this workspace.
        2. Run: `python train.py --epochs 10`
        3. Once training finishes, reload this page to analyze leaves!
        """)
    else:
        with st.spinner("Analyzing leaf pathology..."):
            # Preprocess
            img_resized = image.resize((256, 256))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)[0]
            pred_idx = int(np.argmax(predictions))
            confidence = float(predictions[pred_idx]) * 100
            
            raw_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
            info = metadata_map.get(raw_class, {
                "display_name": raw_class,
                "crop": "Plant",
                "status": "Unknown",
                "pathogen": "N/A",
                "recommendation": "Consult a local agricultural extension specialist."
            })
            
            status = info.get("status", "Unknown")
            status_badge = f'<span class="status-healthy">🟢 {status}</span>' if status == "Healthy" else f'<span class="status-diseased">🔴 {status}</span>'
            
            st.markdown(f"### Diagnosis: **{info.get('display_name', raw_class)}**")
            st.markdown(status_badge, unsafe_allow_html=True)
            
            st.write(f"**Confidence Score:** `{confidence:.1f}%`")
            st.progress(min(1.0, confidence / 100.0))
            
            st.markdown(f"""
            <div class="card-box">
                <h4>📋 Clinical Insights & Care Guide</h4>
                <p><strong>Target Crop:</strong> {info.get('crop', 'N/A')}</p>
                <p><strong>Pathogen / Causative Agent:</strong> {info.get('pathogen', 'N/A')}</p>
                <p><strong>Recommended Treatment:</strong><br>{info.get('recommendation', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 Candidates
            st.write("---")
            st.write("#### Top Predictions Breakdown")
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            for idx in top_3_indices:
                c_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                c_disp = metadata_map.get(c_name, {}).get("display_name", c_name)
                c_conf = float(predictions[idx]) * 100
                st.write(f"• **{c_disp}**: {c_conf:.1f}%")
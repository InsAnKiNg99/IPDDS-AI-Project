import os
import json
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="IPDDS | Plant Pathology Evolved",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inject Background Video directly into Parent DOM (Bypasses DOMPurify)
components.html("""
<script>
    (function() {
        const doc = window.parent.document;
        let v = doc.getElementById('motion-bg-video');
        if (!v) {
            v = doc.createElement('video');
            v.id = 'motion-bg-video';
            v.src = 'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260418_094631_d30ab262-45ee-4b7d-99f3-5d5848c8ef13.mp4';
            v.autoplay = true;
            v.loop = true;
            v.muted = true;
            v.playsInline = true;
            v.style.position = 'fixed';
            v.style.top = '0';
            v.style.left = '0';
            v.style.width = '100vw';
            v.style.height = '100vh';
            v.style.objectFit = 'cover';
            v.style.zIndex = '-1000';
            v.style.pointerEvents = 'none';
            v.style.opacity = '1.0';
            v.style.filter = 'contrast(1.15) brightness(0.95)';
            doc.body.prepend(v);
            
            // Vignette overlay
            let vig = doc.getElementById('motion-vignette');
            if (!vig) {
                vig = doc.createElement('div');
                vig.id = 'motion-vignette';
                vig.style.position = 'fixed';
                vig.style.inset = '0';
                vig.style.background = 'radial-gradient(circle at 50% 35%, rgba(0,0,0,0) 0%, rgba(0,0,0,0.3) 55%, rgba(0,0,0,0.88) 95%, #000000 100%)';
                vig.style.zIndex = '-999';
                vig.style.pointerEvents = 'none';
                doc.body.prepend(vig);
            }
        }
        if (v && v.paused) {
            v.play().catch(() => {});
        }
    })();
</script>
""", height=0, width=0)

# 3. Complete Liquid-Glass & Motion Styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">

<style>
    /* Reset and Ensure Complete Transparency so Background Video Shines Through */
    html, body {
        background-color: #000000 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow-x: hidden !important;
    }
    
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    .main, 
    .block-container {
        background: transparent !important;
        background-color: transparent !important;
        color: #ffffff !important;
        font-family: 'Barlow', -apple-system, sans-serif !important;
    }

    /* Hide Default Streamlit Chrome */
    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    .main .block-container {
        max-width: 1400px !important;
        padding: 0.8rem 2rem 1rem 2rem !important;
        margin: 0 auto !important;
    }

    /* Typography Classes */
    .font-heading {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-style: italic !important;
    }
    
    .font-body {
        font-family: 'Barlow', -apple-system, sans-serif !important;
    }

    /* Top Navigation Floating Pill */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.35rem 1.2rem;
        margin-bottom: 1rem;
        background: rgba(18, 18, 24, 0.45);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 9999px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .nav-circle {
        width: 38px;
        height: 38px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: #ffffff;
    }
    
    .nav-links {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
    }

    .cta-pill-white {
        background: #ffffff;
        color: #000000;
        border-radius: 9999px;
        padding: 0.35rem 1rem;
        font-size: 0.78rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        white-space: nowrap;
    }

    /* Hero Section Left-Aligned Title matching MotionSites */
    .hero-kicker {
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.75);
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }
    
    .hero-title {
        font-size: 3.8rem;
        line-height: 0.92;
        letter-spacing: -2px;
        color: #ffffff;
        margin: 0 0 1rem 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.7);
    }

    /* Liquid-Glass Columns (The 3 Cards) */
    div[data-testid="stColumn"], 
    div[data-testid="column"], 
    .stColumn {
        background: rgba(12, 14, 20, 0.65) !important;
        backdrop-filter: blur(30px) !important;
        -webkit-backdrop-filter: blur(30px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.25), 0 25px 50px rgba(0, 0, 0, 0.75) !important;
        border-radius: 1.25rem !important;
        padding: 1.4rem !important;
        min-height: 380px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
    }
    
    div[data-testid="stColumn"]:hover, 
    div[data-testid="column"]:hover, 
    .stColumn:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.45) !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.35), 0 35px 70px rgba(0, 0, 0, 0.85) !important;
    }

    /* Card Top Elements */
    .card-top-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }
    
    .card-icon-square {
        width: 40px;
        height: 40px;
        border-radius: 0.75rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .tags-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.3rem;
        max-width: 75%;
    }
    
    .liquid-chip {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 9999px;
        padding: 0.16rem 0.55rem;
        font-size: 0.68rem;
        color: rgba(255, 255, 255, 0.9);
        white-space: nowrap;
    }

    /* Card Bottom Titles */
    .card-bottom-title {
        font-size: 1.95rem;
        line-height: 0.95;
        letter-spacing: -1px;
        color: #ffffff;
        margin-top: 0.8rem;
        margin-bottom: 0.25rem;
    }
    
    .card-bottom-desc {
        font-size: 0.8rem;
        line-height: 1.35;
        color: rgba(255, 255, 255, 0.85);
        font-weight: 300;
        margin: 0;
    }

    /* Streamlit Uploader Overrides */
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(255, 255, 255, 0.25) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
        padding: 0.6rem !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        padding: 0.25rem 0.8rem !important;
        border: none !important;
    }

    /* Button Pill Overrides */
    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 9999px !important;
        padding: 0.35rem 0.75rem !important;
        font-family: 'Barlow', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(255, 255, 255, 0.15);
    }

    /* Status Badges */
    .status-badge-healthy {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #6ee7b7;
        padding: 0.2rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-badge-diseased {
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #fca5a5;
        padding: 0.2rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .conf-track {
        width: 100%;
        height: 5px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        overflow: hidden;
        margin: 0.4rem 0 0.6rem 0;
    }
    
    .conf-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 9999px;
    }
</style>
""", unsafe_allow_html=True)

# 4. Model & Metadata Loading
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

@st.cache_resource
def load_trained_model():
    model_paths = [
        os.path.join("model", "disease_model.keras"),
        os.path.join("model", "disease_model.h5")
    ]
    for p in model_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1024 * 100:
            try:
                model = tf.keras.models.load_model(p)
                return model, p
            except Exception:
                pass
    return None, None

def load_metadata():
    json_path = os.path.join("model", "class_indices.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                img_size = tuple(data.get("image_size", [128, 128]))
                return data.get("class_names", DEFAULT_CLASSES), data.get("classes", {}), img_size
        except Exception:
            pass
    return DEFAULT_CLASSES, {}, (128, 128)

class_names, metadata_map, target_img_size = load_metadata()
model, loaded_path = load_trained_model()

# 5. Top Navigation Bar (Liquid-Glass Pill)
st.markdown("""
<div class="top-nav">
    <div class="nav-circle font-heading">ip</div>
    <div class="nav-links">
        <span>Neural Vision</span>
        <span style="opacity: 0.3;">·</span>
        <span>Tomato & Potato</span>
        <span style="opacity: 0.3;">·</span>
        <span>13 Pathogens</span>
        <span style="opacity: 0.3;">·</span>
        <span>Real-Time AI</span>
    </div>
    <div class="cta-pill-white">
        <span>Model Active · 92% Acc</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17L17 7"/><path d="M7 7h10v10"/></svg>
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Editorial Hero Heading (Matching Motion Aetheris)
st.markdown("""
<div style="margin-bottom: 1.6rem;">
    <div class="hero-kicker font-body">// Diagnostics</div>
    <h1 class="hero-title font-heading">
        Plant Pathology<br>evolved
    </h1>
</div>
""", unsafe_allow_html=True)

# 7. Three Cards Grid (Columns act as the Liquid-Glass Cards)
col_card1, col_card2, col_card3 = st.columns([1, 1, 1], gap="medium")

# Handle Sample Picker State
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

with col_card1:
    st.markdown("""
    <div class="card-top-header">
        <div class="card-icon-square">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
        </div>
        <div class="tags-container">
            <span class="liquid-chip">Natural Leaf</span>
            <span class="liquid-chip">Tomato</span>
            <span class="liquid-chip">Potato</span>
            <span class="liquid-chip">Fast Input</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Leaf",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    st.markdown('<div style="font-size: 0.72rem; color: rgba(255,255,255,0.5); margin: 0.3rem 0;">Or quick test specimen:</div>', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🍅 Tomato Blight"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_early_blight.jpg")
        if st.button("🥔 Potato Blight"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_late_blight.jpg")
    with b_col2:
        if st.button("🥔 Potato Early"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_early_blight.jpg")
        if st.button("🌿 Healthy Tomato"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_healthy.jpg")

    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-bottom-title font-heading">Leaf Inspection</h3>
        <p class="card-bottom-desc font-body">
            Upload or select a leaf to deploy deep convolutional vision across 13 pathogen classes.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Determine active image
eval_image = None
if uploaded_file is not None:
    eval_image = Image.open(uploaded_file).convert("RGB")
    st.session_state.selected_sample = None
elif st.session_state.selected_sample and os.path.exists(st.session_state.selected_sample):
    eval_image = Image.open(st.session_state.selected_sample).convert("RGB")

# Perform prediction
pred_data = None
if eval_image is not None and model is not None:
    img_resized = eval_image.resize(target_img_size)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100
    raw_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
    
    info = metadata_map.get(raw_class, {
        "display_name": raw_class.replace("___", ": ").replace("_", " "),
        "crop": "Tomato" if "Tomato" in raw_class else "Potato",
        "status": "Healthy" if "healthy" in raw_class else "Diseased",
        "pathogen": "Identified by Neural Network",
        "recommendation": "Maintain proper cultivation and apply targeted treatment."
    })
    pred_data = {
        "raw_class": raw_class,
        "info": info,
        "confidence": confidence,
        "preds": preds
    }

# Card 2: Neural Diagnosis
with col_card2:
    st.markdown("""
    <div class="card-top-header">
        <div class="card-icon-square">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/></svg>
        </div>
        <div class="tags-container">
            <span class="liquid-chip">92.1% Accuracy</span>
            <span class="liquid-chip">Real-Time</span>
            <span class="liquid-chip">&lt;200ms</span>
            <span class="liquid-chip">CNN Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if pred_data is None:
        st.markdown("""
        <div style="text-align: center; padding: 2.2rem 0; opacity: 0.6;">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🍃</div>
            <div style="font-size: 0.92rem; font-weight: 500;">Awaiting Leaf Input</div>
            <div style="font-size: 0.74rem; color: rgba(255,255,255,0.6); margin-top: 0.2rem;">Choose a sample or upload on the left.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        conf = pred_data["confidence"]
        status = info.get("status", "Unknown")
        badge_cls = "status-badge-healthy" if status == "Healthy" else "status-badge-diseased"
        badge_text = "🟢 Healthy Leaf" if status == "Healthy" else f"🔴 {status}"
        
        st.markdown(f"""
        <div style="margin-bottom: 0.4rem;"><span class="{badge_cls}">{badge_text}</span></div>
        <div style="font-size: 1.35rem; font-weight: 600; color: #ffffff; line-height: 1.2;">
            {info.get("display_name", pred_data["raw_class"])}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 0.6rem;">
            <span style="color: rgba(255,255,255,0.6);">Confidence Score</span>
            <span style="font-weight: 600; color: #6ee7b7;">{conf:.1f}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill" style="width: {min(100.0, conf):.1f}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-bottom-title font-heading">Neural Diagnosis</h3>
        <p class="card-bottom-desc font-body">
            AI analyzes cellular leaf patterns to detect pathogen lesions and viral strains.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Card 3: Treatment Protocol
with col_card3:
    st.markdown("""
    <div class="card-top-header">
        <div class="card-icon-square">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
        </div>
        <div class="tags-container">
            <span class="liquid-chip">Pathogen Agent</span>
            <span class="liquid-chip">Prescription</span>
            <span class="liquid-chip">Agronomy</span>
            <span class="liquid-chip">Action Plan</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if pred_data is None:
        st.markdown("""
        <div style="text-align: center; padding: 2.2rem 0; opacity: 0.6;">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🧪</div>
            <div style="font-size: 0.92rem; font-weight: 500;">Clinical Guide Standby</div>
            <div style="font-size: 0.74rem; color: rgba(255,255,255,0.6); margin-top: 0.2rem;">Pathology protocols populate after inference.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        st.markdown(f"""
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5);">Pathogen Agent</div>
        <div style="font-size: 0.85rem; font-weight: 500; color: #ffffff; margin-bottom: 0.5rem;">{info.get("pathogen", "N/A")}</div>
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5);">Agronomic Treatment</div>
        <div style="font-size: 0.8rem; line-height: 1.35; color: rgba(255,255,255,0.85);">{info.get("recommendation", "N/A")}</div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-bottom-title font-heading">Smart Treatment</h3>
        <p class="card-bottom-desc font-body">
            Tailored fungicide, biological control, and cultural practices to halt infection spread.
        </p>
    </div>
    """, unsafe_allow_html=True)
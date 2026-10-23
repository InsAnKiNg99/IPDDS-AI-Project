import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="IPDDS | Plant Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper to load images locally as base64 for instant, 100% offline-ready display
def get_base64_data_uri(image_path, fallback_url):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/webp;base64,{encoded}"
        except Exception:
            pass
    return fallback_url

FALLBACK_BASE_URL = "https://images.higgs.ai/?default=1&output=webp&url=https%3A%2F%2Fd8j0ntlcm91z4.cloudfront.net%2Fuser_38xzZboKViGWJOttwIXH07lWA1P%2Fhf_20260609_201152_bba90a12-bf12-459f-91f0-51f237dbaf3b.png&w=1920&q=85"
FALLBACK_DISEASED_URL = "https://images.higgs.ai/?default=1&output=webp&url=https%3A%2F%2Fd8j0ntlcm91z4.cloudfront.net%2Fuser_38xzZboKViGWJOttwIXH07lWA1P%2Fhf_20260609_195923_b0ba8ace-1d1d-4f2c-9a28-1ab84b330680.png&w=1920&q=85"

# Local plant flowering rock (base) and decayed rock (reveal)
BASE_IMG_URI = get_base64_data_uri(os.path.join("samples", "bg_image_2.webp"), FALLBACK_BASE_URL)
DISEASED_IMG_URI = get_base64_data_uri(os.path.join("samples", "bg_image_1.webp"), FALLBACK_DISEASED_URL)

# 2. Inject Cursor Spotlight Reveal Engine (Seamless 60 FPS LERP, NO Red Circle)
components.html(f"""
<script>
(function() {{
    const doc = window.parent.document;
    
    // Clean up any old red rings or stale elements
    ['motion-bg-video', 'motion-vignette', 'ipdds-spotlight-ring'].forEach(id => {{
        const el = doc.getElementById(id);
        if (el) el.remove();
    }});
    
    // 1. Base Layer (Lush healthy flowering botanical plant rock)
    let baseLayer = doc.getElementById('ipdds-base-bg');
    if (!baseLayer) {{
        baseLayer = doc.createElement('div');
        baseLayer.id = 'ipdds-base-bg';
        baseLayer.style.position = 'fixed';
        baseLayer.style.top = '0';
        baseLayer.style.left = '0';
        baseLayer.style.width = '100vw';
        baseLayer.style.height = '100vh';
        baseLayer.style.backgroundPosition = 'center';
        baseLayer.style.backgroundSize = 'cover';
        baseLayer.style.backgroundRepeat = 'no-repeat';
        baseLayer.style.zIndex = '0';
        baseLayer.style.pointerEvents = 'none';
        doc.body.prepend(baseLayer);
    }}
    baseLayer.style.backgroundImage = 'url("{BASE_IMG_URI}")';
    
    // 2. Reveal Layer (Diseased/blight rock revealed seamlessly under cursor)
    let revealLayer = doc.getElementById('ipdds-reveal-bg');
    if (!revealLayer) {{
        revealLayer = doc.createElement('div');
        revealLayer.id = 'ipdds-reveal-bg';
        revealLayer.style.position = 'fixed';
        revealLayer.style.top = '0';
        revealLayer.style.left = '0';
        revealLayer.style.width = '100vw';
        revealLayer.style.height = '100vh';
        revealLayer.style.backgroundPosition = 'center';
        revealLayer.style.backgroundSize = 'cover';
        revealLayer.style.backgroundRepeat = 'no-repeat';
        revealLayer.style.zIndex = '1';
        revealLayer.style.pointerEvents = 'none';
        revealLayer.style.maskRepeat = 'no-repeat';
        revealLayer.style.webkitMaskRepeat = 'no-repeat';
        doc.body.prepend(revealLayer);
    }}
    revealLayer.style.backgroundImage = 'url("{DISEASED_IMG_URI}")';

    // 3. Smooth Cursor Tracking with LERP (Pure spotlight mask, NO red circle)
    let mouse = {{ x: -999, y: -999 }};
    let smooth = {{ x: -999, y: -999 }};
    const SPOTLIGHT_R = 270;

    function onMouseMove(e) {{
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    }}

    doc.removeEventListener('mousemove', doc._ipddsMouseMove);
    doc._ipddsMouseMove = onMouseMove;
    doc.addEventListener('mousemove', onMouseMove);

    function loop() {{
        smooth.x += (mouse.x - smooth.x) * 0.12;
        smooth.y += (mouse.y - smooth.y) * 0.12;
        
        if (revealLayer && smooth.x > -500) {{
            const maskVal = `radial-gradient(circle ${{SPOTLIGHT_R}}px at ${{smooth.x}}px ${{smooth.y}}px, black 0%, black 35%, rgba(0,0,0,0.7) 55%, rgba(0,0,0,0.25) 75%, transparent 100%)`;
            revealLayer.style.maskImage = maskVal;
            revealLayer.style.webkitMaskImage = maskVal;
        }}
        
        requestAnimationFrame(loop);
    }}
    
    if (!doc._ipddsLoopRunning) {{
        doc._ipddsLoopRunning = true;
        requestAnimationFrame(loop);
    }}
}})();
</script>
""", height=0, width=0)

# 3. Typography & Responsive Layout CSS System (Clean, Never Clips, Interactive Rock)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@1,400;1,500;1,600&display=swap" rel="stylesheet">

<style>
    /* Clean Viewport Reset */
    *, *::before, *::after {
        box-sizing: border-box !important;
    }
    
    html, body {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 100vh !important;
        background-color: #030708 !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
    }
    
    /* App Layer Sits at z-index 10 (Transparent over the botanical rock at z-index 0/1) */
    .stApp {
        background: transparent !important;
        background-color: transparent !important;
        color: #ffffff !important;
        position: relative !important;
        z-index: 10 !important;
        min-height: 100vh !important;
    }
    
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    section.main {
        background: transparent !important;
        background-color: transparent !important;
        overflow-x: hidden !important;
    }

    /* Hide Streamlit UI Chrome */
    #MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Safe Iframe Handling: Off-screen instead of display:none so browser executes JavaScript */
    [data-testid="stCustomComponentV1"], iframe {
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0.01 !important;
        pointer-events: none !important;
        border: none !important;
    }
    
    /* Centered Main Block Container with Balanced Margins */
    .main .block-container {
        max-width: 1320px !important;
        padding: 1.25rem 2rem 2.5rem 2rem !important;
        margin: 0 auto !important;
    }

    /* Typography Classes */
    .font-playfair {
        font-family: 'Playfair Display', Georgia, serif !important;
    }
    
    .font-inter {
        font-family: 'Inter', sans-serif !important;
    }

    /* Top Floating Navigation Bar */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 1.4rem;
        background: rgba(14, 18, 26, 0.62);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 9999px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 10px 30px rgba(0,0,0,0.6);
        margin-bottom: 1.25rem;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    
    .nav-logo-icon {
        width: 32px;
        height: 32px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .nav-wordmark {
        font-size: 1.35rem;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    
    .nav-center-pill {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.82);
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
        gap: 0.4rem;
        white-space: nowrap;
    }

    /* Hero Heading Section (Clean Project Typography) */
    .hero-container {
        margin-bottom: 1.5rem;
        text-align: left;
    }
    
    .hero-kicker {
        font-size: 0.78rem;
        font-weight: 600;
        color: #34d399;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    
    .hero-title {
        font-size: 2.75rem;
        line-height: 1.05;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -1.2px;
        margin: 0 0 0.5rem 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.8);
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.78);
        max-width: 760px;
        line-height: 1.5;
        margin: 0;
    }

    /* Interactive Bottom Three Liquid-Glass Cards */
    div[data-testid="stHorizontalBlock"] {
        gap: 1.25rem !important;
        align-items: stretch !important;
    }

    div[data-testid="stColumn"], 
    div[data-testid="column"], 
    .stColumn {
        background: rgba(12, 16, 24, 0.76) !important;
        backdrop-filter: blur(40px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 20px 45px rgba(0, 0, 0, 0.75) !important;
        border-radius: 1.25rem !important;
        padding: 1.35rem !important;
        min-height: 420px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        transition: transform 0.25s ease, border-color 0.25s ease !important;
    }
    
    div[data-testid="stColumn"]:hover, 
    div[data-testid="column"]:hover, 
    .stColumn:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.38) !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.3), 0 25px 55px rgba(0, 0, 0, 0.85) !important;
    }

    div[data-testid="stColumn"] > div, 
    div[data-testid="column"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        gap: 0.65rem !important;
    }

    .card-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .card-icon-square {
        width: 36px;
        height: 36px;
        border-radius: 0.7rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .tags-cloud {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.3rem;
        max-width: 80%;
    }
    
    .pill-tag {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 9999px;
        padding: 0.15rem 0.55rem;
        font-size: 0.68rem;
        color: rgba(255, 255, 255, 0.85);
        white-space: nowrap;
    }

    /* Card Bottom Titles */
    .card-title {
        font-size: 1.45rem;
        line-height: 1.1;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0.6rem 0 0.2rem 0;
    }
    
    .card-desc {
        font-size: 0.78rem;
        line-height: 1.4;
        color: rgba(255, 255, 255, 0.75);
        font-weight: 300;
        margin: 0;
    }

    /* File Uploader Clean Liquid Style */
    [data-testid="stFileUploader"] {
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px dashed rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        padding: 0.6rem 0.8rem !important;
    }
    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        padding: 0.25rem 0.85rem !important;
        border: none !important;
    }

    /* Button Pill Overrides */
    .stButton > button {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 9999px !important;
        padding: 0.35rem 0.6rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
        white-space: nowrap !important;
        margin: 0 !important;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.24) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Diagnostic Badges */
    .badge-healthy {
        background: rgba(16, 185, 129, 0.22);
        border: 1px solid rgba(16, 185, 129, 0.55);
        color: #6ee7b7;
        padding: 0.2rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-diseased {
        background: rgba(239, 68, 68, 0.22);
        border: 1px solid rgba(239, 68, 68, 0.55);
        color: #fca5a5;
        padding: 0.2rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .conf-bar-track {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.12);
        border-radius: 9999px;
        overflow: hidden;
        margin: 0.4rem 0 0.6rem 0;
    }
    
    .conf-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 9999px;
    }

    .leaf-preview-box {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 0.45rem 0.65rem;
        margin-top: 0.4rem;
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

def resolve_trained_model_path():
    model_paths = [
        os.path.join("model", "disease_model.keras"),
        os.path.join("model", "disease_model.h5")
    ]
    for p in model_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1024 * 100:
            return p
    return None

@st.cache_resource
def load_trained_model(model_path, model_modified_ns):
    if model_path is None:
        return None, None
    return tf.keras.models.load_model(model_path), model_path

@st.cache_resource
def load_crop_model(model_path, model_modified_ns):
    if model_path is None:
        return None
    return tf.keras.models.load_model(model_path)

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
model_path = resolve_trained_model_path()
model_modified_ns = os.stat(model_path).st_mtime_ns if model_path else 0
model, loaded_path = load_trained_model(model_path, model_modified_ns)
crop_model_path = os.path.join("model", "crop_type_model.keras")
crop_model_modified_ns = os.stat(crop_model_path).st_mtime_ns if os.path.exists(crop_model_path) else 0
crop_model = load_crop_model(crop_model_path if crop_model_modified_ns else None, crop_model_modified_ns)
if model is not None and model.output_shape[-1] != len(class_names):
    raise ValueError("The disease model output count does not match model/class_indices.json.")
if crop_model is not None and crop_model.output_shape[-1] != 2:
    raise ValueError("The crop model must output Potato and Tomato probabilities.")

TOMATO_CROP_CONFIDENCE_THRESHOLD = 0.8

# 5. Top Navigation Bar (Fixed Liquid Pill)
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-logo-icon font-playfair">IP</div>
        <span class="nav-wordmark font-playfair italic">IPDDS</span>
    </div>
    <div class="nav-center-pill">
        <span>Tomato & Potato Crops</span>
        <span style="opacity: 0.3;">·</span>
        <span>13 Disease Classes</span>
        <span style="opacity: 0.3;">·</span>
        <span>Neural Inference</span>
    </div>
    <div class="cta-pill-white">
        <span>Active Model · 92% Acc</span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17L17 7"/><path d="M7 7h10v10"/></svg>
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Hero Heading Section (Clear, Project-Specific Text)
st.markdown("""
<div class="hero-container">
    <div class="hero-kicker">Intelligent Plant Disease Detection System</div>
    <h1 class="hero-title">
        <span class="font-playfair italic">AI Pathology</span> for Potato & Tomato Crops
    </h1>
    <p class="hero-subtitle">
        Neural foliar scanner detecting early blight, late blight, bacterial lesions, and viral infections in real-time with tailored agronomic treatment protocols.
    </p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "selected_label" not in st.session_state:
    st.session_state.selected_label = None

# 7. Interactive Three Liquid-Glass Cards Grid
col_c1, col_c2, col_c3 = st.columns([1, 1, 1], gap="medium")

# Card 1: Leaf Specimen Input
with col_c1:
    st.markdown("""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
        </div>
        <div class="tags-cloud">
            <span class="pill-tag">Tomato</span>
            <span class="pill-tag">Potato</span>
            <span class="pill-tag">Upload</span>
            <span class="pill-tag">Sample</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.85rem; font-weight: 500; margin-bottom: 0.3rem;'>Upload Leaf Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Leaf Image", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        key="leaf_uploader"
    )
    crop_override = st.selectbox(
        "Crop identification",
        ["Auto-detect", "Potato (manual)", "Tomato (manual)"],
        help="Auto-detect uses the trained crop model. Select a crop manually if the automatic result is incorrect."
    )
    
    st.markdown("<div style='font-size: 0.85rem; font-weight: 500; margin: 0.6rem 0 0.35rem 0;'>Or Select Benchmark Leaf</div>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🍅 Tomato Blight", key="btn_tb"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_early_blight.jpg")
            st.session_state.selected_label = "Tomato Early Blight (Demo)"
        if st.button("🥔 Potato Blight", key="btn_pb"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_late_blight.jpg")
            st.session_state.selected_label = "Potato Late Blight (Demo)"
    with b_col2:
        if st.button("🥔 Potato Early", key="btn_pe"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_early_blight.jpg")
            st.session_state.selected_label = "Potato Early Blight (Demo)"
        if st.button("🌿 Healthy Leaf", key="btn_th"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_healthy.jpg")
            st.session_state.selected_label = "Tomato Healthy Leaf (Demo)"

    # Resolve active image
    eval_image = None
    source_name = None
    if uploaded_file is not None:
        eval_image = Image.open(uploaded_file).convert("RGB")
        source_name = uploaded_file.name
    elif st.session_state.selected_sample and os.path.exists(st.session_state.selected_sample):
        eval_image = Image.open(st.session_state.selected_sample).convert("RGB")
        source_name = st.session_state.selected_label or os.path.basename(st.session_state.selected_sample)

    if eval_image is not None:
        st.markdown(f"""
        <div class="leaf-preview-box">
            <span style="font-size: 1.1rem;">📷</span>
            <div style="font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                <span style="font-weight: 600; color: #34d399;">Active Leaf:</span> {source_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-title font-playfair italic">Specimen Ingestion</h3>
        <p class="card-desc">Upload a high-resolution field leaf photograph or click a demo sample to evaluate.</p>
    </div>
    """, unsafe_allow_html=True)

# Run Inference if Image is available
pred_data = None
if eval_image is not None and model is not None:
    img_resized = eval_image.resize(target_img_size)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    if crop_model is not None:
        crop_preds = crop_model.predict(img_array, verbose=0)[0]
        crop_idx = 1 if crop_preds[1] >= TOMATO_CROP_CONFIDENCE_THRESHOLD else 0
        detected_crop = ("Potato", "Tomato")[crop_idx]
        crop_confidence = float(crop_preds[crop_idx]) * 100
    else:
        crop_class_indices = [
            index for index, class_name in enumerate(class_names)
            if class_name.startswith("Potato")
        ]
        tomato_class_indices = [
            index for index, class_name in enumerate(class_names)
            if class_name.startswith("Tomato")
        ]
        potato_probability = float(np.sum(preds[crop_class_indices]))
        tomato_probability = float(np.sum(preds[tomato_class_indices]))
        detected_crop = "Potato" if potato_probability >= tomato_probability else "Tomato"
        crop_confidence = max(potato_probability, tomato_probability) * 100

    crop_name = detected_crop if crop_override == "Auto-detect" else crop_override.split(" ", 1)[0]
    if crop_override != "Auto-detect":
        crop_confidence = None
    crop_class_indices = [
        index for index, class_name in enumerate(class_names)
        if class_name.startswith(crop_name)
    ]
    if not crop_class_indices:
        raise ValueError(f"No disease classes are configured for selected crop: {crop_name}")
    crop_probability = float(np.sum(preds[crop_class_indices]))
    if crop_probability <= 0:
        raise ValueError(f"The disease model returned no probability for selected crop: {crop_name}")
    crop_specific_preds = np.zeros_like(preds)
    crop_specific_preds[crop_class_indices] = preds[crop_class_indices] / crop_probability
    preds = crop_specific_preds

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100
    raw_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"

    info = metadata_map.get(raw_class)
    if info is None or not info.get("recommendation"):
        raise KeyError(f"Missing disease and treatment metadata for predicted class: {raw_class}")
    info = {**info, "crop": crop_name}
    pred_data = {
        "raw_class": raw_class,
        "info": info,
        "confidence": confidence,
        "crop_confidence": crop_confidence,
        "crop_selection": crop_override,
        "preds": preds
    }

# Card 2: Neural Diagnosis
with col_c2:
    if (
        pred_data is not None
        and pred_data["crop_selection"] == "Auto-detect"
        and pred_data["crop_confidence"] < 70
    ):
        st.warning("Crop identification is uncertain. Confirm Potato or Tomato using the selector before relying on the disease result.")
    st.markdown("""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/></svg>
        </div>
        <div class="tags-cloud">
            <span class="pill-tag">92% Acc</span>
            <span class="pill-tag">&lt;150ms</span>
            <span class="pill-tag">Real-Time</span>
            <span class="pill-tag">CNN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if pred_data is None:
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 0.5rem; opacity: 0.7;">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🍃</div>
            <div style="font-size: 0.95rem; font-weight: 500;">Awaiting Leaf Input</div>
            <div style="font-size: 0.78rem; color: rgba(255,255,255,0.65); margin-top: 0.2rem;">
                Upload a photo or choose a benchmark sample on the left to run inference.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        conf = pred_data["confidence"]
        status = info.get("status", "Unknown")
        badge_cls = "badge-healthy" if status == "Healthy" else "badge-diseased"
        badge_text = "🟢 Healthy Crop" if status == "Healthy" else "🔴 Pathogen Detected"
        
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem;"><span class="{badge_cls}">{badge_text}</span></div>
        <div style="font-size: 1.35rem; font-weight: 600; color: #ffffff; line-height: 1.2;">
            {info.get("display_name", pred_data["raw_class"])}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 0.75rem;">
            <span style="color: rgba(255,255,255,0.65);">Confidence Score</span>
            <span style="font-weight: 600; color: #34d399;">{conf:.1f}%</span>
        </div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width: {min(100.0, conf):.1f}%;"></div>
        </div>
        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.6); margin-top: 0.35rem;">
            Crop: <strong style="color: #ffffff;">{info["crop"]}{f" ({pred_data['crop_confidence']:.1f}% confidence)" if pred_data["crop_confidence"] is not None else " (manual selection)"}</strong> · Model Input: 128×128 RGB
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-title font-playfair italic">Neural Diagnosis</h3>
        <p class="card-desc">Deep convolutional network evaluates lesion morphology and foliar patterns.</p>
    </div>
    """, unsafe_allow_html=True)

# Card 3: Treatment Protocol
with col_c3:
    st.markdown("""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
        </div>
        <div class="tags-cloud">
            <span class="pill-tag">Pathogen</span>
            <span class="pill-tag">Treatment</span>
            <span class="pill-tag">Agronomy</span>
            <span class="pill-tag">Remedy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if pred_data is None:
        st.markdown("""
        <div style="text-align: center; padding: 2.5rem 0.5rem; opacity: 0.7;">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🧪</div>
            <div style="font-size: 0.95rem; font-weight: 500;">Clinical Guide Standby</div>
            <div style="font-size: 0.78rem; color: rgba(255,255,255,0.65); margin-top: 0.2rem;">
                Pathogen profile and agronomic remedies will appear upon disease diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        st.markdown(f"""
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5);">Pathogen Agent</div>
        <div style="font-size: 0.92rem; font-weight: 600; color: #ffffff; margin-bottom: 0.6rem;">{info.get("pathogen", "N/A")}</div>
        <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5);">Agronomic Prescription</div>
        <div style="font-size: 0.82rem; line-height: 1.45; color: rgba(255,255,255,0.88);">{info["recommendation"]}</div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-title font-playfair italic">Treatment Protocol</h3>
        <p class="card-desc">Targeted bio-fungicides, copper sprays, and cultural practices to halt disease spread.</p>
    </div>
    """, unsafe_allow_html=True)
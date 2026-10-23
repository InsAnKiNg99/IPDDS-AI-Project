import os
import json
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration (Strict Single-Page No-Scroll Layout)
st.set_page_config(
    page_title="IPDDS | Plant Pathology Evolved",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Botanical Asset URLs (Lush Flowering Rock Base + Diseased Rock Reveal)
BG_IMAGE_BASE = "https://images.higgs.ai/?default=1&output=webp&url=https%3A%2F%2Fd8j0ntlcm91z4.cloudfront.net%2Fuser_38xzZboKViGWJOttwIXH07lWA1P%2Fhf_20260609_201152_bba90a12-bf12-459f-91f0-51f237dbaf3b.png&w=1920&q=85"
BG_IMAGE_DISEASED = "https://images.higgs.ai/?default=1&output=webp&url=https%3A%2F%2Fd8j0ntlcm91z4.cloudfront.net%2Fuser_38xzZboKViGWJOttwIXH07lWA1P%2Fhf_20260609_195923_b0ba8ace-1d1d-4f2c-9a28-1ab84b330680.png&w=1920&q=85"

# 2. Inject Cursor Spotlight Reveal Engine into Parent Window (60 FPS LERP)
components.html(f"""
<script>
(function() {{
    const doc = window.parent.document;
    
    // Clean up any stale elements from previous renders
    ['motion-bg-video', 'motion-vignette', 'ipdds-base-bg', 'ipdds-reveal-bg', 'ipdds-spotlight-ring'].forEach(id => {{
        const el = doc.getElementById(id);
        if (el) el.remove();
    }});
    
    // 1. Base Layer (Lush healthy flowering botanical rock)
    const baseLayer = doc.createElement('div');
    baseLayer.id = 'ipdds-base-bg';
    baseLayer.style.position = 'fixed';
    baseLayer.style.inset = '0';
    baseLayer.style.backgroundImage = 'url("{BG_IMAGE_BASE}")';
    baseLayer.style.backgroundPosition = 'center';
    baseLayer.style.backgroundSize = 'cover';
    baseLayer.style.backgroundRepeat = 'no-repeat';
    baseLayer.style.zIndex = '-999';
    baseLayer.style.pointerEvents = 'none';
    baseLayer.className = 'hero-zoom';
    doc.body.prepend(baseLayer);
    
    // 2. Reveal Layer (Diseased / blight infected rock revealed under cursor)
    const revealLayer = doc.createElement('div');
    revealLayer.id = 'ipdds-reveal-bg';
    revealLayer.style.position = 'fixed';
    revealLayer.style.inset = '0';
    revealLayer.style.backgroundImage = 'url("{BG_IMAGE_DISEASED}")';
    revealLayer.style.backgroundPosition = 'center';
    revealLayer.style.backgroundSize = 'cover';
    revealLayer.style.backgroundRepeat = 'no-repeat';
    revealLayer.style.zIndex = '-998';
    revealLayer.style.pointerEvents = 'none';
    revealLayer.style.maskRepeat = 'no-repeat';
    revealLayer.style.webkitMaskRepeat = 'no-repeat';
    doc.body.prepend(revealLayer);

    // 3. Spotlight Reticle Ring
    const ring = doc.createElement('div');
    ring.id = 'ipdds-spotlight-ring';
    ring.style.position = 'fixed';
    ring.style.top = '0';
    ring.style.left = '0';
    ring.style.width = '500px';
    ring.style.height = '500px';
    ring.style.borderRadius = '50%';
    ring.style.border = '1px dashed rgba(239, 68, 68, 0.45)';
    ring.style.boxShadow = '0 0 35px rgba(239, 68, 68, 0.18), inset 0 0 25px rgba(239, 68, 68, 0.12)';
    ring.style.zIndex = '-997';
    ring.style.pointerEvents = 'none';
    ring.style.transition = 'opacity 0.25s ease';
    ring.style.opacity = '0';
    
    // Scanner Label inside Reticle
    const badge = doc.createElement('div');
    badge.innerText = 'PATHOLOGY SCANNER // BLIGHT REVEAL';
    badge.style.position = 'absolute';
    badge.style.bottom = '14px';
    badge.style.left = '50%';
    badge.style.transform = 'translateX(-50%)';
    badge.style.fontSize = '8.5px';
    badge.style.letterSpacing = '1.8px';
    badge.style.color = 'rgba(254, 202, 202, 0.9)';
    badge.style.fontFamily = 'monospace';
    badge.style.whiteSpace = 'nowrap';
    ring.appendChild(badge);
    
    doc.body.prepend(ring);

    // 4. Smooth Cursor Tracking with LERP (60 FPS)
    let mouse = {{ x: -999, y: -999 }};
    let smooth = {{ x: -999, y: -999 }};
    const SPOTLIGHT_R = 250;

    function onMouseMove(e) {{
        mouse.x = e.clientX;
        mouse.y = e.clientY;
        if (ring) ring.style.opacity = '1';
    }}

    doc.removeEventListener('mousemove', doc._ipddsMouseMove);
    doc._ipddsMouseMove = onMouseMove;
    doc.addEventListener('mousemove', onMouseMove);

    function loop() {{
        smooth.x += (mouse.x - smooth.x) * 0.12;
        smooth.y += (mouse.y - smooth.y) * 0.12;
        
        if (revealLayer && smooth.x > -500) {{
            const maskVal = `radial-gradient(circle ${{SPOTLIGHT_R}}px at ${{smooth.x}}px ${{smooth.y}}px, black 0%, black 38%, rgba(0,0,0,0.75) 58%, rgba(0,0,0,0.35) 76%, rgba(0,0,0,0.08) 90%, transparent 100%)`;
            revealLayer.style.maskImage = maskVal;
            revealLayer.style.webkitMaskImage = maskVal;
        }}
        
        if (ring && smooth.x > -500) {{
            ring.style.transform = `translate3d(${{smooth.x - 250}}px, ${{smooth.y - 250}}px, 0)`;
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

# 3. Typography & Strict Single-Screen Zero-Scroll CSS System
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@1,400;1,500;1,600&display=swap" rel="stylesheet">

<style>
    /* Strict Zero-Scroll Single Viewport (Never Scrolls) */
    *, *::before, *::after {
        box-sizing: border-box !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    
    *::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    html, body {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        overflow-y: hidden !important;
        background-color: #000000 !important;
        margin: 0 !important;
        padding: 0 !important;
        position: fixed !important;
        width: 100vw !important;
    }
    
    .stApp {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        overflow-y: hidden !important;
        background: transparent !important;
        background-color: transparent !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    section.main, 
    .block-container {
        background: transparent !important;
        background-color: transparent !important;
        overflow: hidden !important;
        overflow-y: hidden !important;
    }

    /* Hide Streamlit Default UI Chrome */
    #MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Collapse Streamlit Iframe Wrappers */
    [data-testid="stCustomComponentV1"], iframe {
        display: none !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }
    
    /* Main Layout Viewport Constraint */
    .main .block-container {
        max-width: 1400px !important;
        padding: 0.8rem 2rem 0.8rem 2rem !important;
        margin: 0 auto !important;
        height: 100vh !important;
        max-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    /* Typography Classes */
    .font-playfair {
        font-family: 'Playfair Display', Georgia, serif !important;
    }
    
    .font-inter {
        font-family: 'Inter', sans-serif !important;
    }

    /* Entrance Animations */
    @keyframes heroReveal {
        from { opacity: 0; transform: translateY(22px) scale(0.98); filter: blur(6px); }
        to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }
    @keyframes heroFadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes heroZoom {
        from { transform: scale(1.06); }
        to { transform: scale(1); }
    }
    .hero-anim {
        animation-fill-mode: forwards;
        animation-timing-function: cubic-bezier(0.16, 1, 0.3, 1);
    }
    .hero-reveal {
        animation-name: heroReveal;
        animation-duration: 1.1s;
    }
    .hero-fade {
        animation-name: heroFadeUp;
        animation-duration: 0.9s;
    }
    .hero-zoom {
        animation: heroZoom 2.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Top Floating Navigation Bar */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.35rem 1.25rem;
        background: rgba(14, 16, 22, 0.55);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 9999px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 10px 30px rgba(0,0,0,0.6);
        z-index: 50;
        margin-bottom: 0.2rem;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    
    .nav-logo-icon {
        width: 30px;
        height: 30px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .nav-wordmark {
        font-size: 1.25rem;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    
    .nav-center-pill {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.78rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.82);
    }

    .cta-pill-white {
        background: #ffffff;
        color: #000000;
        border-radius: 9999px;
        padding: 0.3rem 0.95rem;
        font-size: 0.75rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        white-space: nowrap;
    }

    /* Hero Heading Section (Compact, Non-Intrusive) */
    .hero-container {
        margin-top: 0.1rem;
        margin-bottom: 0.4rem;
        pointer-events: none;
    }
    
    .hero-kicker {
        font-size: 0.75rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.75);
        letter-spacing: 1px;
        margin-bottom: 0.1rem;
    }
    
    .hero-h1 {
        line-height: 0.92;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.85);
    }
    
    .hero-line1 {
        display: block;
        font-size: 2.8rem;
        color: #ffffff;
        letter-spacing: -1.5px;
    }
    
    .hero-line2 {
        display: block;
        font-size: 2.8rem;
        font-weight: 300;
        color: #ffffff;
        letter-spacing: -1.8px;
        margin-top: -0.15rem;
    }

    /* Interactive Bottom Three Liquid-Glass Cards */
    div[data-testid="stHorizontalBlock"] {
        margin-top: auto !important;
        margin-bottom: 0.2rem !important;
        gap: 1.2rem !important;
    }

    div[data-testid="stColumn"], 
    div[data-testid="column"], 
    .stColumn {
        background: rgba(14, 16, 22, 0.72) !important;
        backdrop-filter: blur(40px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.22), 0 25px 50px rgba(0, 0, 0, 0.8) !important;
        border-radius: 1.25rem !important;
        padding: 1rem 1.15rem !important;
        height: 310px !important;
        max-height: 310px !important;
        min-height: 310px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        transition: transform 0.25s ease, border-color 0.25s ease !important;
        overflow: hidden !important;
    }
    
    div[data-testid="stColumn"]:hover, 
    div[data-testid="column"]:hover, 
    .stColumn:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.45) !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.35), 0 30px 60px rgba(0, 0, 0, 0.9) !important;
    }

    div[data-testid="stColumn"] > div, 
    div[data-testid="column"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        gap: 0.35rem !important;
    }

    .card-top-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 0.3rem;
    }
    
    .card-icon-square {
        width: 34px;
        height: 34px;
        border-radius: 0.65rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .tags-cloud {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.25rem;
        max-width: 75%;
    }
    
    .pill-tag {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 9999px;
        padding: 0.1rem 0.45rem;
        font-size: 0.64rem;
        color: rgba(255, 255, 255, 0.85);
        white-space: nowrap;
    }

    /* Card Bottom Titles */
    .card-title {
        font-size: 1.55rem;
        line-height: 0.95;
        letter-spacing: -0.6px;
        color: #ffffff;
        margin: 0.4rem 0 0.15rem 0;
    }
    
    .card-desc {
        font-size: 0.72rem;
        line-height: 1.3;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
        margin: 0;
    }

    /* File Uploader Compact Pill Override */
    [data-testid="stFileUploader"] {
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 0.25rem 0.5rem !important;
        min-height: unset !important;
    }
    [data-testid="stFileUploader"] small {
        display: none !important;
    }
    [data-testid="stFileUploader"] span {
        font-size: 0.7rem !important;
    }
    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        font-size: 0.68rem !important;
        padding: 0.15rem 0.65rem !important;
        border: none !important;
    }

    /* Button Pill Overrides */
    .stButton > button {
        background: rgba(255, 255, 255, 0.07) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 9999px !important;
        padding: 0.22rem 0.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.68rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
        white-space: nowrap !important;
        margin: 0 !important;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.22) !important;
        border-color: rgba(255, 255, 255, 0.55) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Diagnostic Badges */
    .badge-healthy {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #6ee7b7;
        padding: 0.12rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.66rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-diseased {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
        padding: 0.12rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.66rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .conf-bar-track {
        width: 100%;
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
        overflow: hidden;
        margin: 0.3rem 0 0.4rem 0;
    }
    
    .conf-bar-fill {
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

# 5. Top Navigation Bar (Fixed Liquid Pill)
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-logo-icon font-playfair">ip</div>
        <span class="nav-wordmark font-playfair italic">IPDDS</span>
    </div>
    <div class="nav-center-pill">
        <span>Neural Pathology</span>
        <span style="opacity: 0.3;">·</span>
        <span>Tomato & Potato</span>
        <span style="opacity: 0.3;">·</span>
        <span>13 Disease Classes</span>
        <span style="opacity: 0.3;">·</span>
        <span>Real-Time Inference</span>
    </div>
    <div class="cta-pill-white">
        <span>Active Model · 92% Acc</span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17L17 7"/><path d="M7 7h10v10"/></svg>
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Hero Heading Section (Staggered Blur-Rise Entrance)
st.markdown("""
<div class="hero-container">
    <div class="hero-kicker hero-anim hero-fade" style="animation-delay: 0.15s;">// AI LEAF PATHOLOGY REVEAL ENGINE</div>
    <h1 class="hero-h1">
        <span class="hero-line1 font-playfair italic hero-anim hero-reveal" style="animation-delay: 0.25s;">Layers hold</span>
        <span class="hero-line2 font-inter hero-anim hero-reveal" style="animation-delay: 0.42s;">tales of plant health</span>
    </h1>
</div>
""", unsafe_allow_html=True)

# 7. Interactive Three Liquid-Glass Cards Grid
col_c1, col_c2, col_c3 = st.columns([1, 1, 1], gap="small")

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# Card 1: Leaf Specimen Input
with col_c1:
    st.markdown("""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
        </div>
        <div class="tags-cloud">
            <span class="pill-tag">Tomato</span>
            <span class="pill-tag">Potato</span>
            <span class="pill-tag">Upload</span>
            <span class="pill-tag">Sample</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Leaf", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🍅 Tomato Blight", key="btn_tb"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_early_blight.jpg")
        if st.button("🥔 Potato Blight", key="btn_pb"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_late_blight.jpg")
    with b_col2:
        if st.button("🥔 Potato Early", key="btn_pe"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_early_blight.jpg")
        if st.button("🌿 Healthy Leaf", key="btn_th"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_healthy.jpg")

    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-title font-playfair italic">Leaf Inspection</h3>
        <p class="card-desc">Select a botanical sample or upload leaf to analyze pathology.</p>
    </div>
    """, unsafe_allow_html=True)

# Active Image Resolution & Prediction
eval_image = None
if uploaded_file is not None:
    eval_image = Image.open(uploaded_file).convert("RGB")
    st.session_state.selected_sample = None
elif st.session_state.selected_sample and os.path.exists(st.session_state.selected_sample):
    eval_image = Image.open(st.session_state.selected_sample).convert("RGB")

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
with col_c2:
    st.markdown("""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/></svg>
        </div>
        <div class="tags-cloud">
            <span class="pill-tag">92% Acc</span>
            <span class="pill-tag">&lt;200ms</span>
            <span class="pill-tag">Real-Time</span>
            <span class="pill-tag">CNN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if pred_data is None:
        st.markdown("""
        <div style="text-align: center; padding: 1.2rem 0; opacity: 0.65;">
            <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">🍃</div>
            <div style="font-size: 0.82rem; font-weight: 500;">Awaiting Leaf Input</div>
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6); margin-top: 0.15rem;">Select a sample or upload on the left.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        conf = pred_data["confidence"]
        status = info.get("status", "Unknown")
        badge_cls = "badge-healthy" if status == "Healthy" else "badge-diseased"
        badge_text = "🟢 Healthy Leaf" if status == "Healthy" else f"🔴 {status}"
        
        st.markdown(f"""
        <div style="margin-bottom: 0.25rem;"><span class="{badge_cls}">{badge_text}</span></div>
        <div style="font-size: 1.15rem; font-weight: 600; color: #ffffff; line-height: 1.15;">
            {info.get("display_name", pred_data["raw_class"])}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-top: 0.4rem;">
            <span style="color: rgba(255,255,255,0.6);">Confidence Score</span>
            <span style="font-weight: 600; color: #6ee7b7;">{conf:.1f}%</span>
        </div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width: {min(100.0, conf):.1f}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-title font-playfair italic">Neural Diagnosis</h3>
        <p class="card-desc">Convolutional network scans cellular patterns to identify pathogen strains.</p>
    </div>
    """, unsafe_allow_html=True)

# Card 3: Treatment Protocol
with col_c3:
    st.markdown("""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
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
        <div style="text-align: center; padding: 1.2rem 0; opacity: 0.65;">
            <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">🧪</div>
            <div style="font-size: 0.82rem; font-weight: 500;">Clinical Guide Standby</div>
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6); margin-top: 0.15rem;">Treatment protocols populate after diagnosis.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        st.markdown(f"""
        <div style="font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5);">Pathogen Agent</div>
        <div style="font-size: 0.8rem; font-weight: 500; color: #ffffff; margin-bottom: 0.35rem;">{info.get("pathogen", "N/A")}</div>
        <div style="font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5);">Agronomic Treatment</div>
        <div style="font-size: 0.73rem; line-height: 1.3; color: rgba(255,255,255,0.85);">{info.get("recommendation", "N/A")}</div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: auto;">
        <h3 class="card-title font-playfair italic">Treatment Protocol</h3>
        <p class="card-desc">Targeted bio-fungicides and cultural practices to halt disease spread.</p>
    </div>
    """, unsafe_allow_html=True)
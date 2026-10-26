import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


st.set_page_config(
    page_title="IPDDS | Plant Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


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


BASE_IMG_URI = get_base64_data_uri(os.path.join("samples", "bg_image_2.webp"), FALLBACK_BASE_URL)
DISEASED_IMG_URI = get_base64_data_uri(os.path.join("samples", "bg_image_1.webp"), FALLBACK_DISEASED_URL)


components.html(f"""
<script>
(function() {{
    const doc = window.parent.document;
    
    ['motion-bg-video', 'motion-vignette', 'ipdds-spotlight-ring'].forEach(id => {{
        const el = doc.getElementById(id);
        if (el) el.remove();
    }});
    
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


st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@1,400;1,500;1,600&display=swap" rel="stylesheet">

<style>
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
    
    .stApp {
        background: transparent !important;
        background-color: transparent !important;
        color: #ffffff !important;
        position: relative !important;
        z-index: 10 !important;
        min-height: 100vh !important;
    }
    
    /* 1. Completely collapse and hide Streamlit's default header */
    header[data-testid="stHeader"],
    header,
    [data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0px !important;
        margin: 0px !important;
    }

    #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0px !important;
        margin: 0px !important;
    }

    /* 2. Strip all top padding and margins from main layout containers */
    .main, 
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stMain"],
    section.main,
    .block-container,
    div[data-testid="block-container"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        top: 0px !important;
    }

    /* Collapse empty element containers and iframes from layout flow */
    div[data-testid="element-container"]:has(iframe),
    div[data-testid="element-container"]:has([data-testid="stCustomComponentV1"]),
    [data-testid="stCustomComponentV1"], 
    iframe {
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        min-height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        opacity: 0.01 !important;
        pointer-events: none !important;
        border: none !important;
    }
    
    .main .block-container,
    [data-testid="block-container"],
    div[data-testid="stAppViewBlockContainer"] {
        max-width: 1320px !important;
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }

    .font-playfair {
        font-family: 'Playfair Display', Georgia, serif !important;
    }
    
    .font-inter {
        font-family: 'Inter', sans-serif !important;
    }

    /* 3. Ensure the custom navbar container has no top margin/offset */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 1.4rem;
        background: rgba(14, 18, 26, 0.75);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 9999px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 10px 30px rgba(0,0,0,0.6);
        margin-top: 0rem !important;
        top: 0px !important;
        margin-bottom: 1.15rem !important;
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

    div[data-testid="stHorizontalBlock"] {
        gap: 1.25rem !important;
        align-items: stretch !important;
    }

    /* Primary Cards (top-level columns only) */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"], 
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        background: rgba(12, 16, 24, 0.76) !important;
        backdrop-filter: blur(40px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2), 0 20px 45px rgba(0, 0, 0, 0.75) !important;
        border-radius: 1.25rem !important;
        padding: 1.35rem !important;
        min-height: 440px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        transition: transform 0.25s ease, border-color 0.25s ease !important;
    }
    
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:hover, 
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.38) !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.3), 0 25px 55px rgba(0, 0, 0, 0.85) !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div, 
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        gap: 0.65rem !important;
    }

    /* Reset nested button columns inside primary cards */
    div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
        margin-bottom: 0.2rem !important;
    }
    div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        min-height: auto !important;
        display: block !important;
        transform: none !important;
    }
    div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:hover {
        transform: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div {
        height: auto !important;
        display: block !important;
        gap: 0.35rem !important;
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

    [data-testid="stFileUploader"] {
        margin-bottom: 0.4rem !important;
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

    /* Glassmorphic Selectbox styling */
    [data-testid="stSelectbox"] {
        margin-bottom: 0.4rem !important;
    }
    [data-testid="stSelectbox"] label p {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: rgba(255, 255, 255, 0.85) !important;
        margin-bottom: 0.2rem !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
        border-color: rgba(255, 255, 255, 0.4) !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #ffffff !important;
        background: transparent !important;
    }
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"], 
    ul[role="listbox"] {
        background: #0d131a !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.85) !important;
    }
    ul[role="listbox"] li {
        color: #ffffff !important;
        background: transparent !important;
        font-size: 0.82rem !important;
    }
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {
        background: rgba(52, 211, 153, 0.18) !important;
        color: #34d399 !important;
    }

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


DEFAULT_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
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
    return tf.keras.models.load_model(model_path, compile=False), model_path

@st.cache_resource
def load_crop_model(model_path, model_modified_ns):
    if model_path is None:
        return None
    return tf.keras.models.load_model(model_path, compile=False)

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

def validate_plant_leaf(image: Image.Image) -> tuple[bool, str]:
    """
    Validates whether an input image is a genuine plant/leaf specimen.
    Rejects out-of-distribution inputs (black/white images, noise, selfies, objects, non-leaf photos).
    Returns (is_valid, reason_message).
    """
    if image is None:
        return False, "No specimen provided."

    np_img = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w, _ = np_img.shape
    if h < 32 or w < 32:
        return False, "Specimen resolution is too low."

    # 1. Texture & Signal Variance (reject blank / flat / solid color images)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    channel_std = float(np.std(np_img))
    if lap_var < 35.0 or channel_std < 12.0:
        return False, "Image lacks foliage texture or contrast."

    # 2. Total Variation (reject synthetic random/speckle noise)
    diff_x = np.mean(np.abs(np_img[:, 1:, :].astype(float) - np_img[:, :-1, :].astype(float)))
    diff_y = np.mean(np.abs(np_img[1:, :, :].astype(float) - np_img[:-1, :, :].astype(float)))
    tv = (diff_x + diff_y) / 2.0
    if tv > 58.0:
        return False, "Synthetic noise or high-frequency non-organic pattern detected."

    # 3. Foliage & Chlorophyll Color Analysis (HSV + Excess Green Index)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Chlorophyll green spectrum (Hue 28-88, Sat >= 35, Val >= 30)
    green_mask = (hue >= 28) & (hue <= 88) & (sat >= 35) & (val >= 30)
    # Lesion / necrosis spectrum (Hue 14-27, Sat >= 40, Val >= 30)
    yellow_brown_mask = (hue >= 14) & (hue < 28) & (sat >= 40) & (val >= 30)
    foliage_mask = green_mask | yellow_brown_mask

    green_ratio = float(np.mean(green_mask))
    foliage_ratio = float(np.mean(foliage_mask))

    # Excess Green Index: 2G - R - B
    r = np_img[:, :, 0].astype(np.float32)
    g = np_img[:, :, 1].astype(np.float32)
    b = np_img[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    exg_pos = float(np.mean(exg > 5.0))

    # 4. Spatial Clustering (verify connected anatomical leaf body)
    foliage_uint8 = foliage_mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foliage_uint8)
    max_cluster_area = 0
    if num_labels > 1:
        max_cluster_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    cluster_ratio = max_cluster_area / float(h * w)

    is_plant = (
        (green_ratio >= 0.08 or (foliage_ratio >= 0.15 and green_ratio >= 0.04) or exg_pos >= 0.12)
        and cluster_ratio >= 0.03
    )

    if not is_plant:
        return False, "Specimen lacks plant foliage pigments or anatomical leaf structure."

    return True, "Valid leaf specimen."

class_names, metadata_map, target_img_size = load_metadata()
model_path = resolve_trained_model_path()
model_modified_ns = os.stat(model_path).st_mtime_ns if model_path else 0
model, loaded_path = load_trained_model(model_path, model_modified_ns)
crop_model_path = os.path.join("model", "crop_type_model.keras")
crop_model_modified_ns = os.stat(crop_model_path).st_mtime_ns if os.path.exists(crop_model_path) else 0
crop_model = load_crop_model(crop_model_path if crop_model_modified_ns else None, crop_model_modified_ns)

if model is not None and model.output_shape[-1] != len(class_names):
    st.error("The disease model output count does not match model/class_indices.json.")
if crop_model is not None and crop_model.output_shape[-1] != 2:
    st.error("The crop model must output Potato and Tomato probabilities.")

TOMATO_CROP_CONFIDENCE_THRESHOLD = 0.8


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


if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "selected_label" not in st.session_state:
    st.session_state.selected_label = None


col_c1, col_c2 = st.columns([1, 1.4], gap="medium")


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
            <span class="pill-tag">Benchmark</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.82rem; font-weight: 500; margin-bottom: 0.25rem;'>Upload Leaf Image</div>", unsafe_allow_html=True)
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
    
    st.markdown("<div style='font-size: 0.82rem; font-weight: 500; margin: 0.5rem 0 0.3rem 0;'>Or Select Benchmark Leaf</div>", unsafe_allow_html=True)
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

    eval_image = None
    source_name = None
    if uploaded_file is not None:
        try:
            eval_image = Image.open(uploaded_file).convert("RGB")
            source_name = uploaded_file.name
        except Exception:
            eval_image = None
            source_name = None
    elif st.session_state.selected_sample and os.path.exists(st.session_state.selected_sample):
        try:
            eval_image = Image.open(st.session_state.selected_sample).convert("RGB")
            source_name = st.session_state.selected_label or os.path.basename(st.session_state.selected_sample)
        except Exception:
            eval_image = None
            source_name = None

    is_valid_specimen = False
    validation_error = None
    if eval_image is not None:
        is_valid_specimen, val_msg = validate_plant_leaf(eval_image)
        if not is_valid_specimen:
            validation_error = "No valid leaf or plant detected in the photo. Please upload a clear image of a tomato or potato leaf."

    if eval_image is not None:
        if validation_error:
            st.error(validation_error)
        else:
            st.markdown(f"""
            <div class="leaf-preview-box">
                <span style="font-size: 1.1rem;">📷</span>
                <div style="font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    <span style="font-weight: 600; color: #34d399;">Active Specimen:</span> {source_name}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: auto; padding-top: 0.5rem;">
        <h3 class="card-title font-playfair italic">Specimen Ingestion</h3>
        <p class="card-desc">Upload a high-resolution field leaf photograph or click a benchmark sample to evaluate.</p>
    </div>
    """, unsafe_allow_html=True)


pred_data = None
inference_error = None

if eval_image is not None and is_valid_specimen and model is not None:
    try:
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
            potato_probability = float(np.sum(preds[crop_class_indices])) if crop_class_indices else 0.0
            tomato_probability = float(np.sum(preds[tomato_class_indices])) if tomato_class_indices else 0.0
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
            inference_error = f"No disease classes are configured for selected crop: {crop_name}"
        else:
            crop_probability = float(np.sum(preds[crop_class_indices]))
            if crop_probability <= 0:
                inference_error = f"The disease model returned no probability for selected crop: {crop_name}"
            else:
                crop_specific_preds = np.zeros_like(preds)
                crop_specific_preds[crop_class_indices] = preds[crop_class_indices] / crop_probability
                preds = crop_specific_preds

                pred_idx = int(np.argmax(preds))
                confidence = float(preds[pred_idx]) * 100
                raw_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"

                info = metadata_map.get(raw_class)
                if info is None or not info.get("recommendation"):
                    inference_error = f"Missing clinical metadata for predicted class: {raw_class}"
                else:
                    info = {**info, "crop": crop_name}
                    pred_data = {
                        "raw_class": raw_class,
                        "info": info,
                        "confidence": confidence,
                        "crop_confidence": crop_confidence,
                        "crop_selection": crop_override,
                        "preds": preds
                    }
    except Exception as e:
        inference_error = f"Diagnostic processing notice: {str(e)}"


with col_c2:
    st.markdown("""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        <div class="tags-cloud">
            <span class="pill-tag">92% Acc</span>
            <span class="pill-tag">&lt;150ms</span>
            <span class="pill-tag">Dual-Stage CNN</span>
            <span class="pill-tag">Clinical Guide</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if inference_error:
        st.error(inference_error)

    if pred_data is None:
        st.markdown("""
        <div style="text-align: center; padding: 3.2rem 1rem; opacity: 0.72;">
            <div style="font-size: 2.5rem; margin-bottom: 0.4rem;">🍃</div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #ffffff;">Awaiting Foliar Specimen</div>
            <div style="font-size: 0.82rem; color: rgba(255,255,255,0.65); max-width: 440px; margin: 0.4rem auto 0 auto; line-height: 1.5;">
                Upload a field leaf photograph or choose a benchmark sample on the left to activate neural diagnosis and clinical remedies.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        conf = pred_data["confidence"]
        status = info.get("status", "Unknown")
        badge_cls = "badge-healthy" if status == "Healthy" else "badge-diseased"
        badge_text = "🟢 Healthy Specimen" if status == "Healthy" else "🔴 Pathogen Detected"
        crop_label = info.get("crop", "Solanaceae")
        crop_conf_str = f" ({pred_data['crop_confidence']:.1f}%)" if pred_data["crop_confidence"] is not None else " (Manual)"

        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.65rem;">
            <div><span class="{badge_cls}">{badge_text}</span></div>
            <span style="font-size: 0.76rem; color: rgba(255, 255, 255, 0.78); background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 9999px; padding: 0.18rem 0.65rem;">
                Host Crop: <strong style="color: #ffffff;">{crop_label}{crop_conf_str}</strong>
            </span>
        </div>

        <div style="font-size: 1.45rem; font-weight: 600; color: #ffffff; line-height: 1.2; margin-bottom: 0.2rem;">
            {info.get("display_name", pred_data["raw_class"])}
        </div>

        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 0.65rem;">
            <span style="color: rgba(255,255,255,0.65);">Pathology Confidence Score</span>
            <span style="font-weight: 600; color: #34d399;">{conf:.1f}%</span>
        </div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width: {min(100.0, conf):.1f}%;"></div>
        </div>

        <div style="border-top: 1px solid rgba(255, 255, 255, 0.12); margin: 0.85rem 0 0.8rem 0;"></div>

        <div style="display: flex; flex-direction: column; gap: 0.65rem;">
            <div>
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5); font-weight: 600;">Causal Pathogen / Agent</div>
                <div style="font-size: 0.92rem; font-weight: 600; color: #e2e8f0; margin-top: 0.15rem;">
                    {info.get("pathogen", "N/A")}
                </div>
            </div>
            <div>
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255,255,255,0.5); font-weight: 600;">Agronomic Prescription & Remediation</div>
                <div style="font-size: 0.83rem; line-height: 1.5; color: rgba(255,255,255,0.92); margin-top: 0.25rem; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 0.65rem 0.85rem;">
                    {info.get("recommendation", "Standard cultivation practices recommended.")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style="margin-top: auto; padding-top: 0.8rem;">
        <h3 class="card-title font-playfair italic">Neural Pathology & Clinical Protocol</h3>
        <p class="card-desc">Foliar lesion pathology coupled directly with agronomic countermeasures.</p>
    </div>
    """, unsafe_allow_html=True)
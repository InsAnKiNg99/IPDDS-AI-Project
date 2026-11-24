import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2


st.set_page_config(
    page_title="IPDDS · Potato and tomato leaf check",
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

def has_leaf_color_distribution(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized_rgb = tf.image.resize(rgb, (128, 128), method="bilinear", antialias=False).numpy()
    hsv = np.asarray(
        Image.fromarray(np.clip(np.rint(resized_rgb), 0, 255).astype(np.uint8)).convert("HSV"),
        dtype=np.uint8
    )
    hue = hsv[:, :, 0].astype(np.float32) * (360.0 / 255.0)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    brightness = hsv[:, :, 2].astype(np.float32) / 255.0
    leaf_colors = (
        (hue >= 35.0)
        & (hue <= 155.0)
        & (saturation >= 0.14)
        & (brightness >= 0.12)
    )
    center_leaf_ratio = float(leaf_colors[32:96, 32:96].mean())
    leaf_brightness = brightness[leaf_colors]
    brightness_std = float(leaf_brightness.std()) if leaf_brightness.size else 0.0
    leaf_color_ratio = float(leaf_colors.mean())
    return (
        0.12 <= leaf_color_ratio <= 0.98
        and center_leaf_ratio >= 0.20
        and brightness_std >= 0.09
    )

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

    let dimLayer = doc.getElementById('ipdds-dim');
    if (!dimLayer) {{
        dimLayer = doc.createElement('div');
        dimLayer.id = 'ipdds-dim';
        dimLayer.style.position = 'fixed';
        dimLayer.style.inset = '0';
        dimLayer.style.background = 'rgba(4, 8, 10, 0.62)';
        dimLayer.style.zIndex = '2';
        dimLayer.style.pointerEvents = 'none';
        doc.body.appendChild(dimLayer);
    }}

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
        
        if (revealLayer && smooth.x > -500 && !doc.body.classList.contains('ipdds-enter')) {{
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

    const root = doc.documentElement;
    if (root.dataset.ipddsEntered === '1') {{
        doc.body.classList.add('ipdds-settled');
    }} else if (root.dataset.ipddsEntered !== 'pending') {{
        root.dataset.ipddsEntered = 'pending';
        doc.body.classList.remove('ipdds-settled');
        window.setTimeout(() => {{
            doc.body.classList.add('ipdds-enter');
            window.setTimeout(() => {{
                doc.body.classList.remove('ipdds-enter');
                doc.body.classList.add('ipdds-settled');
                root.dataset.ipddsEntered = '1';
            }}, 1800);
        }}, 3500);
    }}
}})();
</script>
""", height=0, width=0)


st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

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
        font-family: 'Inter', sans-serif !important;
        font-style: normal !important;
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
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
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
        font-size: 1.05rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.02em;
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
        font-size: 2.15rem;
        line-height: 1.15;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin: 0 0 0.45rem 0;
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
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
        border-radius: 1.25rem !important;
        padding: 1.35rem !important;
        min-height: 440px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        transition: border-color 0.2s ease !important;
    }
    
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:hover, 
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
        border-color: rgba(255, 255, 255, 0.28) !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div, 
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
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

    .panel-title {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #ffffff !important;
        line-height: 1.2 !important;
        margin: 0 0 0.2rem 0 !important;
    }

    .card-title {
        font-size: 1.02rem;
        font-weight: 600;
        line-height: 1.2;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0.35rem 0 0.15rem 0;
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
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    .stButton > button p {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
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
        white-space: normal !important;
        line-height: 1.25 !important;
        min-height: 2.3rem !important;
        margin: 0 0 0.35rem 0 !important;
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
        background: #34d399;
        border-radius: 9999px;
    }

    .conf-bar-fill.is-disease {
        background: #e7b089;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
        width: 100%;
        max-height: 240px;
        object-fit: cover;
    }

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    @keyframes ipdds-from-top {
        from { opacity: 0; transform: translate3d(0, -84px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    @keyframes ipdds-from-bottom {
        from { opacity: 0; transform: translate3d(0, 72px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    @keyframes ipdds-from-left {
        from { opacity: 0; transform: translate3d(-96px, 0, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    @keyframes ipdds-from-right {
        from { opacity: 0; transform: translate3d(96px, 0, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }

    body:not(.ipdds-settled) .top-nav,
    body:not(.ipdds-settled) .hero-container,
    body:not(.ipdds-settled) [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        opacity: 0;
        pointer-events: none;
    }

    body:not(.ipdds-settled) [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        opacity: 1;
        pointer-events: auto;
        transform: none;
        animation: none;
    }

    body.ipdds-enter [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        animation: none;
        opacity: 1;
        transform: none;
    }

    body.ipdds-enter .top-nav,
    body.ipdds-enter .hero-container,
    body.ipdds-enter [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        will-change: transform, opacity;
        backface-visibility: hidden;
    }

    body.ipdds-enter [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    body.ipdds-enter .top-nav {
        animation: ipdds-from-top 1.35s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }

    body.ipdds-enter .hero-container {
        animation: ipdds-from-bottom 1.35s cubic-bezier(0.22, 1, 0.36, 1) 0.1s forwards;
    }

    body.ipdds-enter [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(1) {
        animation: ipdds-from-left 1.35s cubic-bezier(0.22, 1, 0.36, 1) 0.16s forwards;
    }

    body.ipdds-enter [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {
        animation: ipdds-from-right 1.35s cubic-bezier(0.22, 1, 0.36, 1) 0.22s forwards;
    }

    body.ipdds-settled .top-nav,
    body.ipdds-settled .hero-container,
    body.ipdds-settled [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        opacity: 1;
        transform: none;
        animation: none;
        pointer-events: auto;
        will-change: auto;
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

    .empty-state {
        padding: 2.4rem 0.4rem 1.2rem 0;
    }

    .empty-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }

    .empty-copy {
        font-size: 0.86rem;
        line-height: 1.45;
        color: rgba(255, 255, 255, 0.68);
        max-width: 36rem;
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
    with open(json_path, "r", encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)
    if not isinstance(data, dict):
        raise ValueError("model/class_indices.json must contain a JSON object.")
    names = data.get("class_names")
    classes = data.get("classes")
    img_size = data.get("image_size")
    if (
        not isinstance(names, list)
        or len(names) != 13
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("model/class_indices.json must define 13 unique disease labels.")
    if (
        not isinstance(classes, dict)
        or set(classes) != set(names)
        or not all(isinstance(details, dict) for details in classes.values())
    ):
        raise ValueError("model/class_indices.json disease metadata does not match its labels.")
    if (
        not isinstance(img_size, list)
        or len(img_size) != 2
        or not all(type(dimension) is int and dimension > 0 for dimension in img_size)
    ):
        raise ValueError("model/class_indices.json has an invalid image size.")
    return names, classes, tuple(img_size)

def foliage_fraction(image: Image.Image) -> float:
    """Share of pixels with real leaf green. Pale cloth can share a leaf hue and still fail this."""
    rgb = np.asarray(image.convert("RGB"))
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    lit = (red > 12) | (green > 12) | (blue > 12)
    counted = int(lit.sum())
    if counted < 50:
        return 0.0
    foliage = (2 * green - red - blue > 30) & (green > 35) & (green < 245)
    return float(foliage.sum()) / counted

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

    # 3. Leaf pigment. Hue alone accepts khaki and pale green cloth.
    if foliage_fraction(image) < 0.04:
        return False, "Specimen lacks plant foliage pigments or anatomical leaf structure."

    return True, "Valid leaf specimen."

def crop_to_foliage(image):
    """Zoom a small leaf up to the frame. Leave photos that already fill the frame alone."""
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    foliage = (hue >= 8) & (hue <= 100) & (saturation >= 28) & (value >= 30)
    coordinates = np.argwhere(foliage)
    height, width = foliage.shape
    if coordinates.shape[0] < 0.04 * height * width:
        return image
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    box_height = int(bottom - top + 1)
    box_width = int(right - left + 1)
    if box_width * box_height >= 0.55 * height * width:
        return image
    pad_y = int(0.12 * box_height)
    pad_x = int(0.12 * box_width)
    crop_top = max(0, int(top) - pad_y)
    crop_left = max(0, int(left) - pad_x)
    crop_bottom = min(height, int(bottom) + 1 + pad_y)
    crop_right = min(width, int(right) + 1 + pad_x)
    return image.crop((crop_left, crop_top, crop_right, crop_bottom))

def model_input_batch(image, img_size):
    framed = crop_to_foliage(image)
    rgb = np.asarray(framed.convert("RGB"), dtype=np.float32)
    resized = tf.image.resize_with_pad(rgb, int(img_size[0]), int(img_size[1])).numpy()
    return resized[None, ...]

def limit_photo_size(image, max_side=960):
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = max_side / max(width, height)
    if scale < 1:
        return rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
    return rgb

def plant_view_batch(image, img_size):
    """Full photo plus a grid of leaf-sized patches, so a whole plant can be read leaf by leaf."""
    rgb = limit_photo_size(image)
    width, height = rgb.size
    views = [rgb]
    side = int(min(width, height) * 0.52)
    if 140 <= side <= int(min(width, height) * 0.9):
        for top in np.linspace(0, max(height - side, 0), num=3):
            for left in np.linspace(0, max(width - side, 0), num=3):
                views.append(rgb.crop((int(left), int(top), int(left) + side, int(top) + side)))
    else:
        zoom = max(32, int(min(width, height) * 0.62))
        left = max(0, (width - zoom) // 2)
        top = max(0, (height - zoom) // 2)
        views.append(rgb.crop((left, top, left + zoom, top + zoom)))
    frames = []
    for view in views:
        framed = crop_to_foliage(view)
        pixels = np.asarray(framed.convert("RGB"), dtype=np.float32)
        frames.append(tf.image.resize_with_pad(pixels, int(img_size[0]), int(img_size[1])).numpy())
    return np.stack(frames, axis=0)

def _valid_probability_row(row, expected):
    return (
        getattr(row, "shape", None) == (expected,)
        and np.all(np.isfinite(row))
        and np.all((row >= 0) & (row <= 1))
        and np.isclose(float(np.sum(row)), 1.0, atol=0.02)
    )

def diagnose_plant_views(crop_preds, disease_preds, class_names, crop_classes):
    votes = []
    for crop_row, disease_row in zip(crop_preds, disease_preds):
        if not _valid_probability_row(crop_row, 2) or not _valid_probability_row(disease_row, len(class_names)):
            continue
        crop_idx = int(np.argmax(crop_row))
        if float(crop_row[crop_idx]) < 0.60:
            continue
        crop_name = crop_classes[crop_idx]
        class_indices = [index for index, name in enumerate(class_names) if name.startswith(crop_name)]
        crop_total = float(np.sum(disease_row[class_indices]))
        if crop_total <= 0:
            continue
        local = disease_row[class_indices] / crop_total
        best_offset = int(np.argmax(local))
        votes.append({
            "crop": crop_name,
            "crop_conf": float(crop_row[crop_idx]),
            "class_index": class_indices[best_offset],
            "class_conf": float(local[best_offset]),
        })
    if not votes:
        return None

    crop_name = max({vote["crop"] for vote in votes}, key=lambda name: sum(vote["crop"] == name for vote in votes))
    crop_votes = [vote for vote in votes if vote["crop"] == crop_name]
    crop_confidence = float(np.mean([vote["crop_conf"] for vote in crop_votes]))
    crop_share = len(crop_votes) / len(votes)
    if crop_share < 0.5 or crop_confidence < 0.70:
        return {"uncertain_crop": True, "crop_confidence": crop_confidence * 100, "views": len(votes)}

    if len(crop_votes) == 1:
        vote = crop_votes[0]
        chosen_index = vote["class_index"]
        best_mean = vote["class_conf"]
        uncertain = best_mean < 0.35
    elif len(crop_votes) == 2 and crop_votes[0]["class_index"] != crop_votes[1]["class_index"]:
        winner, other = sorted(crop_votes, key=lambda vote: vote["class_conf"], reverse=True)
        chosen_index = winner["class_index"]
        best_mean = winner["class_conf"]
        uncertain = (winner["class_conf"] - other["class_conf"]) < 0.08 or best_mean < 0.35
    else:
        buckets = {}
        for vote in crop_votes:
            buckets.setdefault(vote["class_index"], []).append(vote["class_conf"])
        chosen_index = max(
            buckets,
            key=lambda class_index: (len(buckets[class_index]), float(np.mean(buckets[class_index]))),
        )
        chosen_confs = buckets[chosen_index]
        best_mean = float(np.mean(chosen_confs))
        share = len(chosen_confs) / len(crop_votes)
        is_healthy = "healthy" in class_names[chosen_index].lower()
        needed_share = 0.45 if is_healthy else 0.60
        uncertain = share < needed_share or best_mean < 0.40

    averaged = np.zeros(len(class_names), dtype=np.float32)
    averaged[chosen_index] = best_mean
    return {
        "uncertain_crop": False,
        "raw_class": class_names[chosen_index],
        "confidence": best_mean * 100,
        "crop_name": crop_name,
        "crop_confidence": crop_confidence * 100,
        "uncertain": uncertain,
        "views": len(crop_votes),
        "preds": averaged,
    }

def brown_spot_fraction(image):
    """Brown pixels compared with green pixels. Shadows stay green and are not spots."""
    rgb = np.asarray(limit_photo_size(image, 640).convert("RGB")).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    spots = (red > green + 12) & (red > blue) & (red > 35) & (green < 180)
    foliage = (green > red + 8) & (green > blue) & (green > 40)
    foliage_pixels = int(foliage.sum())
    if foliage_pixels < 50:
        return 1.0
    return float(spots.sum()) / foliage_pixels

def correct_clean_canopy(image, diagnosis, class_names):
    """Late blight on a green canopy with no brown spots is a false alarm."""
    if not diagnosis or diagnosis.get("uncertain_crop"):
        return diagnosis
    raw_class = diagnosis.get("raw_class", "")
    if "healthy" in raw_class.lower() or "Late_blight" not in raw_class:
        return diagnosis
    if brown_spot_fraction(image) >= 0.02:
        return diagnosis
    healthy_name = next(
        name for name in class_names
        if name.startswith(diagnosis["crop_name"]) and "healthy" in name.lower()
    )
    corrected = dict(diagnosis)
    corrected["raw_class"] = healthy_name
    corrected["confidence"] = 78.0
    corrected["uncertain"] = False
    corrected["clean_canopy"] = True
    averaged = np.zeros(len(class_names), dtype=np.float32)
    averaged[class_names.index(healthy_name)] = 0.78
    corrected["preds"] = averaged
    return corrected

class_names, metadata_map, target_img_size = load_metadata()
model_path = resolve_trained_model_path()
model_modified_ns = os.stat(model_path).st_mtime_ns if model_path else 0
model, loaded_path = load_trained_model(model_path, model_modified_ns)
crop_model_path = os.path.join("model", "crop_type_model.keras")
crop_model_modified_ns = os.stat(crop_model_path).st_mtime_ns if os.path.exists(crop_model_path) else 0
crop_model = load_crop_model(crop_model_path if crop_model_modified_ns else None, crop_model_modified_ns)


with open(os.path.join("model", "training_metrics.json"), "r", encoding="utf-8") as metrics_file:
    training_metrics = json.load(metrics_file)
crop_classes = training_metrics.get("crop_classes") if isinstance(training_metrics, dict) else None
if crop_classes != ["Potato", "Tomato"]:
    raise ValueError("model/training_metrics.json must map crop model outputs to Potato, then Tomato.")
training_datasets = training_metrics.get("training_datasets", [])
if not isinstance(training_datasets, list) or not all(
    isinstance(source, str) and source for source in training_datasets
):
    raise ValueError("model/training_metrics.json has an invalid training_datasets list.")
dataset_summary = "PlantVillage" if len(training_datasets) == 1 else (
    f"{len(training_datasets)} PlantVillage sources" if training_datasets else "PlantVillage"
)
if model is not None and model.output_shape[-1] != len(class_names):
    st.error("The disease model output count does not match model/class_indices.json.")
if crop_model is not None and crop_model.output_shape[-1] != 2:
    st.error("The crop model must output Potato and Tomato probabilities.")

    raise ValueError("The crop model must output Potato and Tomato probabilities.")
if model is None or crop_model is None:
    st.error("The disease and crop models must both be available before IPDDS can analyze a photo.")
    st.stop()
for metric_name in ("disease_accuracy", "crop_accuracy"):
    metric_value = training_metrics.get(metric_name)
    if not isinstance(metric_value, (int, float)) or not np.isfinite(metric_value) or not 0 <= metric_value <= 1:
        raise ValueError(f"model/training_metrics.json has an invalid {metric_name} value.")

MIN_CROP_CONFIDENCE = 0.70
CROP_WARNING_THRESHOLD = 0.85
MIN_DISEASE_CONFIDENCE = 0.35


st.markdown(f"""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-logo-icon font-playfair">IP</div>
        <span class="nav-wordmark">IPDDS</span>
    </div>
    <div class="nav-center-pill">
        <span>Potato and tomato</span>
        <span style="opacity: 0.3;">·</span>
        <span>13 diseases</span>
        <span style="opacity: 0.3;">·</span>
        <span>{dataset_summary}</span>
    </div>
    <div class="cta-pill-white">
        <span>{training_metrics["disease_accuracy"] * 100:.1f}% held-out accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-container">
    <div class="hero-kicker">Leaf check · version 1.1</div>
    <h1 class="hero-title">Check a potato or tomato leaf</h1>
    <p class="hero-subtitle">
        Upload one leaf. IPDDS estimates the crop and the most likely condition, then suggests what to do next. Treat the result as a guide and confirm anything important in the field.
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
            <span class="pill-tag">One leaf</span>
            <span class="pill-tag">JPG or PNG</span>
        </div>
    </div>
    <h2 class="panel-title">Photo</h2>
    <p class="card-desc" style="margin-bottom: 0.55rem;">A close photo of a single leaf works best.</p>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.82rem; font-weight: 500; margin-bottom: 0.25rem;'>Upload a photo</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a photo", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        key="leaf_uploader"
    )
    preview_slot = st.empty()

    st.markdown("<div style='font-size: 0.85rem; font-weight: 500; margin: 0.7rem 0 0.35rem 0;'>Or try an example</div>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("Tomato blight", key="btn_tb"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_early_blight.jpg")
            st.session_state.selected_label = "Tomato early blight example"
        if st.button("Potato late", key="btn_pb"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_late_blight.jpg")
            st.session_state.selected_label = "Potato late blight example"
    with b_col2:
        if st.button("Potato early", key="btn_pe"):
            st.session_state.selected_sample = os.path.join("samples", "sample_potato_early_blight.jpg")
            st.session_state.selected_label = "Potato early blight example"
        if st.button("Healthy leaf", key="btn_th"):
            st.session_state.selected_sample = os.path.join("samples", "sample_tomato_healthy.jpg")
            st.session_state.selected_label = "Healthy tomato example"

    eval_image = None
    source_name = None
    if uploaded_file is not None:
        try:
            eval_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            source_name = uploaded_file.name
        except Exception:
            eval_image = None
            source_name = None
    elif st.session_state.selected_sample and os.path.exists(st.session_state.selected_sample):
        try:
            eval_image = ImageOps.exif_transpose(Image.open(st.session_state.selected_sample)).convert("RGB")
            source_name = st.session_state.selected_label or os.path.basename(st.session_state.selected_sample)
        except Exception:
            eval_image = None
            source_name = None

    is_valid_specimen = False
    validation_error = None
    if eval_image is not None:
        is_valid_specimen, _validation_message = validate_plant_leaf(crop_to_foliage(eval_image))
        if not is_valid_specimen:
            validation_error = "This photo does not look like a tomato or potato leaf. Try a closer shot of one leaf."

    with preview_slot.container():
        if eval_image is not None:
            if validation_error:
                st.error(validation_error)
            else:
                st.markdown(f"""
                <div class="leaf-preview-box">
                    <div style="font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        <span style="font-weight: 600; color: #d7e3dc;">Photo</span> · {source_name}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.image(eval_image, use_container_width=True)

    st.markdown("""
    <div style="margin-top: auto; padding-top: 0.35rem;">
        <p class="card-desc">The examples are leaves already in this project. Your own photo replaces them.</p>
    </div>
    """, unsafe_allow_html=True)


pred_data = None
inference_error = None
crop_estimate_uncertain = False
disease_estimate_uncertain = False

if eval_image is not None and is_valid_specimen and model is not None and crop_model is not None:
    try:
        view_batch = plant_view_batch(eval_image, target_img_size)
        crop_preds = crop_model.predict(view_batch, verbose=0)
        disease_preds = model.predict(view_batch, verbose=0)
        diagnosis = diagnose_plant_views(crop_preds, disease_preds, class_names, crop_classes)
        diagnosis = correct_clean_canopy(eval_image, diagnosis, class_names)
        if diagnosis is None:
            raise ValueError("The photo did not produce a usable leaf reading.")
        crop_estimate_uncertain = diagnosis.get("uncertain_crop", False)
        if not crop_estimate_uncertain:
            raw_class = diagnosis["raw_class"]
            leaves_disagree = bool(diagnosis["uncertain"] and diagnosis["views"] > 2)
            single_leaf_too_weak = bool(diagnosis["uncertain"] and diagnosis["views"] <= 2)
            disease_estimate_uncertain = single_leaf_too_weak
            if not single_leaf_too_weak:
                info = metadata_map.get(raw_class)
                if info is None or not info.get("recommendation"):
                    raise KeyError(f"Missing disease and treatment metadata for predicted class: {raw_class}")
                pred_data = {
                    "raw_class": raw_class,
                    "info": {**info, "crop": diagnosis["crop_name"]},
                    "confidence": diagnosis["confidence"],
                    "crop_confidence": diagnosis["crop_confidence"],
                    "preds": diagnosis["preds"],
                    "views": diagnosis["views"],
                    "leaves_disagree": leaves_disagree and not diagnosis.get("clean_canopy"),
                    "clean_canopy": bool(diagnosis.get("clean_canopy")),
                }
    except Exception as error:
        inference_error = f"Could not read this photo: {error}"


with col_c2:
    st.markdown(f"""
    <div class="card-top-row">
        <div class="card-icon-square">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        <div class="tags-cloud">
            <span class="pill-tag">Disease {training_metrics["disease_accuracy"] * 100:.1f}%</span>
            <span class="pill-tag">Crop {training_metrics["crop_accuracy"] * 100:.1f}%</span>
            <span class="pill-tag">{target_img_size[0]}×{target_img_size[1]}</span>
        </div>
    </div>
    <h2 class="panel-title">Result</h2>
    """, unsafe_allow_html=True)

    if inference_error:
        st.error(inference_error)
    elif eval_image is not None and not is_valid_specimen:
        st.warning("No clear potato or tomato leaf was found. Center one leaf and try again.")
    elif crop_estimate_uncertain:
        st.warning("The crop is unclear. Use a photo with one leaf in the middle of the frame.")
    elif disease_estimate_uncertain:
        st.warning("The disease call is too uncertain to trust. Try another clear photo.")
    elif pred_data is not None and pred_data.get("clean_canopy"):
        st.info("No brown or black spotting was visible, so this plant is read as healthy.")
    elif pred_data is not None and pred_data.get("leaves_disagree"):
        st.warning("Leaves in this photo do not all agree. The strongest reading is shown. Check one leaf up close before you act.")
    elif pred_data is not None and pred_data["crop_confidence"] < CROP_WARNING_THRESHOLD * 100:
        st.warning("The crop call is only moderately sure. Look at the plant before you rely on the care note.")

    if pred_data is None and eval_image is not None and not is_valid_specimen:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">That photo was skipped</div>
            <div class="empty-copy">Center one potato or tomato leaf so it fills most of the frame.</div>
        </div>
        """, unsafe_allow_html=True)
    elif pred_data is None and crop_estimate_uncertain:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">Crop not clear enough</div>
            <div class="empty-copy">Move closer so one leaf sits in the middle, then try again.</div>
        </div>
        """, unsafe_allow_html=True)
    elif pred_data is None and disease_estimate_uncertain:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">Disease not clear enough</div>
            <div class="empty-copy">A sharper photo of the spotted or damaged area usually helps.</div>
        </div>
        """, unsafe_allow_html=True)
    elif pred_data is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">No photo yet</div>
            <div class="empty-copy">Upload a leaf, or pick one of the examples on the left.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = pred_data["info"]
        conf = pred_data["confidence"]
        status = info.get("status", "Unknown")
        badge_cls = "badge-healthy" if status == "Healthy" else "badge-diseased"
        badge_text = "Healthy" if status == "Healthy" else "Disease found"
        bar_class = "conf-bar-fill" if status == "Healthy" else "conf-bar-fill is-disease"
        confidence_color = "#34d399" if status == "Healthy" else "#e7b089"
        crop_label = info.get("crop", "Unknown")
        crop_conf_str = f" ({pred_data['crop_confidence']:.1f}%)"
        input_side = int(target_img_size[0])
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.65rem;">
            <div><span class="{badge_cls}">{badge_text}</span></div>
            <span style="font-size: 0.76rem; color: rgba(255, 255, 255, 0.78); background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 9999px; padding: 0.18rem 0.65rem;">
                Crop: <strong style="color: #ffffff;">{crop_label}{crop_conf_str}</strong>
            </span>
        </div>
        <div style="font-size: 1.45rem; font-weight: 600; color: #ffffff; line-height: 1.2; margin-bottom: 0.2rem;">
            {info.get("display_name", pred_data["raw_class"])}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 0.65rem;">
            <span style="color: rgba(255,255,255,0.65);">Confidence</span>
            <span style="font-weight: 600; color: {confidence_color};">{conf:.1f}%</span>
        </div>
        <div class="conf-bar-track">
            <div class="{bar_class}" style="width: {min(100.0, conf):.1f}%;"></div>
        </div>
        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.6); margin-top: 0.35rem;">
            Crop confidence {pred_data['crop_confidence']:.1f}% · checked at {input_side}×{input_side}{"" if pred_data.get("views", 1) <= 1 else f" · {pred_data['views']} leaf areas"}
        </div>
        <div style="border-top: 1px solid rgba(255, 255, 255, 0.12); margin: 0.85rem 0 0.8rem 0;"></div>
        <div style="display: flex; flex-direction: column; gap: 0.65rem;">
            <div>
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: rgba(255,255,255,0.5); font-weight: 600;">Cause</div>
                <div style="font-size: 0.92rem; font-weight: 600; color: #e2e8f0; margin-top: 0.15rem;">
                    {info.get("pathogen", "N/A")}
                </div>
            </div>
            <div>
                <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: rgba(255,255,255,0.5); font-weight: 600;">What to do</div>
                <div style="font-size: 0.83rem; line-height: 1.5; color: rgba(255,255,255,0.92); margin-top: 0.25rem; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 0.65rem 0.85rem;">
                    {info.get("recommendation", "Standard cultivation practices recommended.")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: auto; padding-top: 0.8rem;">
        <p class="card-desc">Care notes are general. Check the plant before you spray or remove it.</p>
    </div>
    """, unsafe_allow_html=True)

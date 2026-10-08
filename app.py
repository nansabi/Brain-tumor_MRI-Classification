import os
import urllib.request

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def densenet_preprocess(x):
    return tf.keras.applications.densenet.preprocess_input(x)


# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
CREAM = "#FCF0DA"
SAGE = "#AEAC78"
GOLD = "#F2C46A"
BROWN = "#4C4541"

# ----------------------------------------------------------------------------
# Global styling
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');

:root {{
    --cream: {CREAM};
    --sage: {SAGE};
    --gold: {GOLD};
    --brown: {BROWN};
    --cream-deep: #F5E3BE;
}}

html, body, [class*="css"], .stApp {{
    font-family: 'DM Sans', sans-serif;
    color: var(--brown);
}}

.stApp {{
    background: var(--cream);
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {{
    visibility: hidden;
    height: 0;
}}

.block-container {{
    max-width: 780px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}}

/* ---------- Headings ---------- */
h1, h2, h3 {{
    font-family: 'Fraunces', serif !important;
    color: var(--brown) !important;
    letter-spacing: -0.01em;
}}

/* ---------- Hero ---------- */
.hero {{
    padding: 0.5rem 0 1.75rem 0;
    border-bottom: 2px solid var(--sage);
    margin-bottom: 2rem;
}}
.hero h1 {{
    font-size: 2.9rem;
    line-height: 1.08;
    font-weight: 600;
    margin: 0 0 0.75rem 0;
    padding: 0;
}}
.hero p {{
    font-size: 1.08rem;
    line-height: 1.65;
    max-width: 56ch;
    margin: 0;
    color: var(--brown);
    opacity: 0.85;
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: var(--brown);
    border-right: 4px solid var(--gold);
}}
[data-testid="stSidebar"] * {{
    color: var(--cream);
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: var(--gold) !important;
}}
.guide-item {{
    padding: 0.75rem 0 0.75rem 0.9rem;
    border-left: 3px solid var(--sage);
    margin-bottom: 0.9rem;
}}
.guide-item b {{
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: var(--gold);
    display: block;
    margin-bottom: 0.15rem;
}}
.guide-item span {{
    font-size: 0.9rem;
    line-height: 1.5;
    opacity: 0.9;
}}
.disclaimer {{
    background: rgba(242, 196, 106, 0.12);
    border: 1px solid var(--gold);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 0.85rem;
    line-height: 1.55;
    margin-top: 1.25rem;
}}
.disclaimer b {{
    color: var(--gold);
}}
[data-testid="stSidebar"] input {{
    background: var(--cream) !important;
    color: var(--brown) !important;
    border-radius: 8px;
}}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] section {{
    background: var(--cream-deep);
    border: 2px dashed var(--sage);
    border-radius: 16px;
    padding: 1.6rem;
    transition: border-color 0.2s ease, background 0.2s ease;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: var(--brown);
    background: #F8E8C6;
}}
[data-testid="stFileUploader"] label p {{
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
}}
[data-testid="stFileUploader"] button {{
    background: var(--brown) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.45rem 1.3rem !important;
    font-weight: 600;
}}
[data-testid="stFileUploader"] button:hover {{
    background: #3A3431 !important;
}}
[data-testid="stFileUploader"] small {{
    color: var(--brown);
    opacity: 0.7;
}}

/* ---------- Image ---------- */
[data-testid="stImage"] img {{
    border-radius: 16px;
    border: 3px solid var(--brown);
    box-shadow: 8px 8px 0 var(--sage);
}}
[data-testid="stImage"] figcaption,
[data-testid="stCaptionContainer"] {{
    color: var(--brown) !important;
    opacity: 0.75;
    text-align: center;
}}

/* ---------- Result card ---------- */
.result-card {{
    background: var(--brown);
    color: var(--cream);
    border-radius: 20px;
    padding: 2rem 2.2rem 1.8rem 2.2rem;
    margin: 2.2rem 0 1.2rem 0;
    border-bottom: 6px solid var(--gold);
}}
.result-card .label {{
    font-size: 0.95rem;
    opacity: 0.75;
    margin-bottom: 0.2rem;
}}
.result-card .verdict {{
    font-family: 'Fraunces', serif;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.1;
    color: var(--gold);
    margin-bottom: 1.4rem;
}}
.result-card .conf-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}}
.result-card .conf-row strong {{
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    color: var(--cream);
}}
.result-card .track {{
    height: 10px;
    background: rgba(252, 240, 218, 0.18);
    border-radius: 999px;
    overflow: hidden;
}}
.result-card .fill {{
    height: 100%;
    background: var(--gold);
    border-radius: 999px;
}}
.result-card .about {{
    margin-top: 1.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(252, 240, 218, 0.2);
    font-size: 0.95rem;
    line-height: 1.6;
    opacity: 0.9;
}}

/* ---------- Notices ---------- */
.notice {{
    border-radius: 12px;
    padding: 0.95rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.55;
    margin: 0.5rem 0 1.5rem 0;
}}
.notice.ok {{
    background: rgba(174, 172, 120, 0.28);
    border-left: 5px solid var(--sage);
}}
.notice.warn {{
    background: rgba(242, 196, 106, 0.35);
    border-left: 5px solid var(--gold);
}}

/* ---------- Section title ---------- */
.section-title {{
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 600;
    margin: 2rem 0 0.6rem 0;
}}

/* ---------- Alerts (st.info / st.error / st.warning) ---------- */
[data-testid="stAlert"] {{
    background: var(--cream-deep);
    border: 1px solid var(--sage);
    border-radius: 12px;
    color: var(--brown);
}}

/* Spinner */
[data-testid="stSpinner"] * {{
    color: var(--brown) !important;
}}

:focus-visible {{
    outline: 3px solid var(--gold) !important;
    outline-offset: 2px;
}}

@media (max-width: 640px) {{
    .hero h1 {{ font-size: 2.1rem; }}
    .result-card .verdict {{ font-size: 2.1rem; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No tumor",
    "pituitary": "Pituitary tumor",
}
CLASS_INFO = {
    "glioma": "A tumor that starts in the brain's glial cells. It often needs prompt specialist attention.",
    "meningioma": "A tumor that grows on the meninges, the membranes that cover the brain.",
    "no_tumor": "No abnormal mass was detected in this scan.",
    "pituitary": "A tumor on the pituitary gland at the base of the brain. Most are benign.",
}
MODEL_FILE = "densenet121_clean.keras"

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("## Classification guide")
st.sidebar.markdown(
    """
<div class="guide-item"><b>Glioma</b><span>Starts in the brain's glial cells and often needs prompt intervention.</span></div>
<div class="guide-item"><b>Meningioma</b><span>Grows on the membranes (meninges) that cover the brain.</span></div>
<div class="guide-item"><b>Pituitary</b><span>Usually benign. Sits on the pituitary gland at the base of the brain.</span></div>
<div class="guide-item"><b>No tumor</b><span>Healthy tissue with no abnormal mass detected.</span></div>
<div class="disclaimer"><b>Medical disclaimer</b><br>
This is an educational demo built on deep learning. It does not provide a medical diagnosis.
Ask a qualified clinician to review any MRI.</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
MODEL_URL = st.sidebar.text_input(
    "Model download URL (optional)",
    value="",
    help="If the model is hosted externally (for example on Hugging Face), paste the direct download link here.",
)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_tumor_model(local_path, download_url=None):
    """Load the CNN once and cache it to keep the app's memory footprint small."""
    if not os.path.exists(local_path):
        if download_url:
            with st.spinner("Downloading the model. This can take a moment..."):
                try:
                    urllib.request.urlretrieve(download_url, local_path)
                except Exception as e:
                    st.error(f"Could not download the model: {e}")
                    return None
        else:
            st.warning(
                f"Model file '{local_path}' was not found. "
                "Place it in the same folder as app.py, or add a download URL in the sidebar."
            )
            return None

    try:
        # compile=False: the model has scaling layers but needs no training state
        return tf.keras.models.load_model(
            local_path,
            compile=False,
            custom_objects={"densenet_preprocess": densenet_preprocess},
        )
    except Exception as e:
        st.error(f"Could not load the Keras model file: {e}")
        return None


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <h1>Brain tumor MRI classifier</h1>
    <p>Upload a brain MRI scan to see the predicted category, how confident the model is,
    and how the probability is spread across all four classes.</p>
</div>
""",
    unsafe_allow_html=True,
)

model = load_tumor_model(MODEL_FILE, download_url=MODEL_URL if MODEL_URL else None)

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
    help="JPG or PNG. The image is resized to 224 x 224 before analysis.",
)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded brain MRI scan", use_container_width=True)

        # Preprocessing matches the Colab pipeline:
        # RGB -> 224x224 -> float32 in [0, 1] -> add batch dimension
        img_rgb = image.convert("RGB")
        img_resized = img_rgb.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        input_batch = np.expand_dims(img_array, axis=0)

        if model is not None:
            with st.spinner("Analyzing MRI features..."):
                preds = model.predict(input_batch, verbose=0)[0]

            best_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[best_idx]
            confidence = float(preds[best_idx]) * 100

            # Result card
            st.markdown(
                f"""
<div class="result-card">
    <div class="label">Predicted category</div>
    <div class="verdict">{DISPLAY_NAMES[pred_class]}</div>
    <div class="conf-row"><span>Model confidence</span><strong>{confidence:.1f}%</strong></div>
    <div class="track"><div class="fill" style="width:{confidence:.1f}%"></div></div>
    <div class="about">{CLASS_INFO[pred_class]}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            # Confidence notice
            if confidence < 70.0:
                st.markdown(
                    '<div class="notice warn"><b>Low confidence.</b> The model is below 70% sure. '
                    "A specialist should review this image carefully.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="notice ok"><b>High confidence.</b> '
                    "The model is strongly leaning toward this category.</div>",
                    unsafe_allow_html=True,
                )

            # Probability chart
            st.markdown('<div class="section-title">Probability by category</div>', unsafe_allow_html=True)

            labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
            values = preds * 100
            colors = [GOLD if i == best_idx else SAGE for i in range(len(CLASS_NAMES))]

            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor(CREAM)
            ax.set_facecolor(CREAM)

            bars = ax.barh(
                labels[::-1],
                values[::-1],
                color=colors[::-1],
                edgecolor=BROWN,
                linewidth=1.2,
                height=0.58,
            )
            for bar, val in zip(bars, values[::-1]):
                ax.text(
                    min(val + 1.5, 88),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=10,
                    fontweight="bold",
                    color=BROWN,
                )

            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", color=BROWN, fontsize=10)
            ax.tick_params(colors=BROWN, labelsize=10, length=0)
            ax.xaxis.grid(True, color=SAGE, alpha=0.35, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(BROWN)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info(
                "The model isn't loaded yet. Make sure the model file is in the app folder, "
                "or paste a download URL in the sidebar."
            )

    except Exception as e:
        st.error(f"Could not process this scan: {e}")
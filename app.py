import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# PART 1: APPLICATION FOUNDATION AND VISUAL IDENTITY
# Why this part exists:
# This section creates first impression and trust.
# Recruiters judge clarity, structure, and UI thinking before model logic.
# Users should instantly understand what the system does without reading docs.
# Dark UI with extreme contrast avoids eye strain and prevents faded text issues.
# This part intentionally runs before any ML logic to set visual consistency.

st.set_page_config(
    page_title="Food Quality Inspection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# GLOBAL UI + TEXT VISIBILITY FIX
# Why this is required:
# Streamlit injects low-opacity styles internally which cause faded text.
# We force brightness, font weight, and opacity globally.
# This guarantees every label, radio text, metric, and paragraph stays readable.

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #020617;
    color: #ffffff !important;
    font-size: 22px;
    font-weight: 900;
}

.main, .block-container {
    background-color: #020617;
    padding-top: 2.2rem;
}

* {
    opacity: 1 !important;
}

/* HEADERS */

.header-box {
    background: linear-gradient(90deg, #1e3a8a, #6d28d9);
    padding: 52px 60px;
    border-radius: 28px;
    margin-bottom: 35px;
    box-shadow: 0 0 45px rgba(56,189,248,0.45);
}

.header-title {
    font-size: 56px;
    font-weight: 950;
    color: #ffffff;
}

.header-subtitle {
    font-size: 24px;
    font-weight: 900;
    color: #f8fafc;
    margin-top: 8px;
}

.header-tagline {
    font-size: 22px;
    font-weight: 900;
    color: #38bdf8;
    margin-top: 10px;
}

/* SECTION TITLES */

.section-title {
    font-size: 32px;
    font-weight: 950;
    color: #ffffff;
    margin-top: 28px;
    margin-bottom: 12px;
}

/* BUTTONS */

.stButton > button {
    background-color: #020617;
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    border-radius: 16px;
    border: 2px solid #38bdf8;
    padding: 16px 0;
    box-shadow: 0 0 26px rgba(56,189,248,0.45);
}

.stButton > button span {
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 900 !important;
}

/* RADIO / UPLOAD TEXT FIX */

label, p, span, div {
    color: #ffffff !important;
    font-weight: 900 !important;
}

/* FILE UPLOADER VISIBILITY */

.stFileUploader label,
.stFileUploader span,
.stFileUploader div {
    color: #ffffff !important;
    font-size: 22px !important;
    font-weight: 900 !important;
}

/* METRICS */

[data-testid="stMetricValue"] {
    font-size: 36px;
    font-weight: 950;
    color: #ffffff;
}

[data-testid="stMetricLabel"] {
    font-size: 20px;
    font-weight: 900;
    color: #e5e7eb;
}

/* GRAPH SIZE CONTROL PREP
   Actual graph will use this small size in Part 2 */

.small-graph {
    max-width: 520px;
    margin-left: auto;
    margin-right: auto;
}

</style>
""", unsafe_allow_html=True)


# MAIN HEADER
# Why this structure:
# Big title establishes authority.
# Subtitle explains ML capability in one glance.
# Tagline converts tech into business value language.

st.markdown("""
<div class="header-box">
    <div class="header-title">Food Quality Inspection System</div>
    <div class="header-subtitle">
        Image-Based Freshness Detection and Shelf-Life Risk Prediction
    </div>
    <div class="header-tagline">
        From food images to confident safety decisions
    </div>
</div>
""", unsafe_allow_html=True)



# SHORT INSTRUCTIONS ONLY (2 LINES AS REQUESTED)
# Why only two lines:
# Users do not read paragraphs.
# Clear, short instructions reduce cognitive load.

st.markdown("""
<p style="font-size:24px;font-weight:900;color:white;">
Upload a food image to check freshness.<br>
The system converts visual quality into clear action guidance.
</p>
""", unsafe_allow_html=True)


# ROLE SELECTION
# Why buttons instead of sidebar:
# Role selection is a core decision, not a setting.
# Keeping it on main screen improves clarity and UX.

st.markdown("<div class='section-title'>Choose Your Role</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    customer_btn = st.button("Customer Mode", use_container_width=True)

with col2:
    shopkeeper_btn = st.button("Shopkeeper Mode", use_container_width=True)


# SESSION STATE ROLE MANAGEMENT
# Why session state:
# Prevents page reload confusion.
# Maintains smooth flow across interactions.

if "role" not in st.session_state:
    st.session_state.role = None

if customer_btn:
    st.session_state.role = "Customer"

if shopkeeper_btn:
    st.session_state.role = "Shopkeeper"

if st.session_state.role == "Customer":
    st.success("Customer mode active")

if st.session_state.role == "Shopkeeper":
    st.success("Shopkeeper mode active")




# PART 2: MODEL INTEGRATION AND CUSTOMER INTERACTION
# Why this part exists:
# ML outputs alone are meaningless to users.
# This layer translates probabilities into human decisions.
# Customer logic focuses on consumption safety.
# Graph is compact to avoid scrolling.
# All outputs are intentionally large and readable.

MODEL_PATH = "food_quality_model.h5"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr)[0][0])
    freshness = round((1 - prob) * 100, 2)
    confidence = round(max(freshness, 100 - freshness), 2)

    if freshness >= 80:
        risk = "Low Risk"
        action = "Safe to consume"
        comment = "Food quality looks fresh and healthy."
        color = "#22c55e"
    elif freshness >= 50:
        risk = "Medium Risk"
        action = "Consume today"
        comment = "Quality declining. Consume within 24 hours."
        color = "#facc15"
    else:
        risk = "High Risk"
        action = "Do not consume"
        comment = "Poor quality detected. Consumption not advised."
        color = "#ef4444"

    return freshness, confidence, risk, action, comment, color

def draw_compact_graph(score, color):
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    labels = ["Today", "Tomorrow", "Day After"]
    values = [score, max(score - 15, 0), max(score - 35, 0)]

    ax.bar(labels, values, color=color)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Freshness", color="white", fontsize=12, fontweight="bold")
    ax.set_title("Freshness Trend (48 Hours)", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    st.pyplot(fig)

# Customer flow
if st.session_state.role == "Customer":

    st.markdown("<div class='section-title'>Customer Food Inspection</div>", unsafe_allow_html=True)

    mode = st.radio(
        "Select input method",
        ["Upload Image", "Use Camera"],
        horizontal=True
    )

    image_source = None

    if mode == "Upload Image":
        image_source = st.file_uploader(
            "Upload food image",
            type=["jpg", "jpeg", "png", "webp"]
        )

    if mode == "Use Camera":
        image_source = st.camera_input("Capture food image")

    if image_source:
        temp_path = "temp_image.png"
        with open(temp_path, "wb") as f:
            f.write(image_source.read())

        st.image(image_source, width=260)

        freshness, confidence, risk, action, comment, color = predict_image(temp_path)

        c1, c2, c3 = st.columns(3)
        c1.metric("Freshness Score", f"{freshness} %")
        c2.metric("Risk Level", risk)
        c3.metric("Confidence", f"{confidence} %")

        st.progress(int(confidence))

        st.markdown(
            f"<h2>Action: {action}</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h3>System Comment: {comment}</h3>",
            unsafe_allow_html=True
        )

        draw_compact_graph(freshness, color)








# PART 3: SHOPKEEPER MODE – BATCH INSPECTION & INVENTORY DECISIONS
# Why this part exists:
# Customers inspect single items, but shopkeepers operate at batch level.
# Real retail decisions involve crates, trays, and daily deliveries.
# This block extends the same ML logic to multiple items at once.
# The model output remains identical, only interpretation changes.
# This separation shows product thinking and role-aware system design.

import pandas as pd
from datetime import datetime

# Sidebar is used ONLY for shopkeeper to reduce main-page clutter

st.markdown("""
<style>

/* SIDEBAR BACKGROUND FIX
   Forces sidebar to use the same dark theme as main app
   Removes Streamlit default light/gray sidebar completely */
section[data-testid="stSidebar"] {
    background-color: #020617 !important;
}

/* SIDEBAR CONTAINER
   Ensures inner sidebar area stays dark and consistent */
section[data-testid="stSidebar"] > div {
    background-color: #020617 !important;
}

/* SIDEBAR TEXT VISIBILITY
   Makes all sidebar text bright white and bold */
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div {
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

/* SIDEBAR RADIO & CHECKBOX LABELS
   Fixes faded option text */
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 900 !important;
}

/* SIDEBAR BUTTONS
   Makes buttons dark with bright text */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #020617 !important;
    color: #ffffff !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 14px !important;
}

/* SIDEBAR FILE UPLOADER TEXT
   Fixes gray upload text */
section[data-testid="stSidebar"] .stFileUploader *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: #ffffff !important;
    font-weight: 900 !important;
}

/* REMOVE ANY FADE EFFECT IN SIDEBAR */
section[data-testid="stSidebar"] * {
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)


if st.session_state.role == "Shopkeeper":

    st.sidebar.markdown(
        "<h2 style='color:white;font-weight:900;'>Shopkeeper Controls</h2>",
        unsafe_allow_html=True
    )

    shopkeeper_input = st.sidebar.radio(
        "Inspection method",
        ["Upload Image Batch", "Use Camera"],
        key="shopkeeper_input_mode"
    )

    st.markdown("<div class='section-title'>Shopkeeper Batch Inspection</div>", unsafe_allow_html=True)

    st.write(
        "Inspect multiple food items together to make inventory decisions. "
        "The system analyzes each item and summarizes overall batch risk."
    )

    batch_images = []

    # Batch upload (multiple images)
    if shopkeeper_input == "Upload Image Batch":
        uploaded_files = st.file_uploader(
            "Upload multiple food images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="shopkeeper_batch_upload"
        )
        if uploaded_files:
            batch_images = uploaded_files

    # Camera-based batch scan (one-by-one, but treated as batch)
    if shopkeeper_input == "Use Camera":
        cam = st.camera_input(
            "Scan food item",
            key="shopkeeper_camera"
        )
        if cam:
            batch_images = [cam]

    # Process batch
    if batch_images:

        os.makedirs("temp_batch", exist_ok=True)
        batch_results = []

        for idx, img in enumerate(batch_images):
            temp_path = f"temp_batch/item_{idx}.png"
            with open(temp_path, "wb") as f:
                f.write(img.read())

            freshness, confidence, risk, _, _, color = predict_image(temp_path)

            # Shopkeeper-specific interpretation
            if risk == "Low Risk":
                action = "Store and sell normally"
            elif risk == "Medium Risk":
                action = "Prioritize sale or discount"
            else:
                action = "Remove from inventory"

            batch_results.append({
                "Item ID": idx + 1,
                "Freshness Score (%)": freshness,
                "Risk Level": risk,
                "Recommended Action": action,
                "Confidence (%)": confidence
            })

        df = pd.DataFrame(batch_results)

        st.markdown("<div class='section-title'>Batch Results</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # Summary metrics
        low = (df["Risk Level"] == "Low Risk").sum()
        mid = (df["Risk Level"] == "Medium Risk").sum()
        high = (df["Risk Level"] == "High Risk").sum()

        s1, s2, s3 = st.columns(3)
        s1.metric("Low Risk Items", low)
        s2.metric("Medium Risk Items", mid)
        s3.metric("High Risk Items", high)

        # Batch-level decision logic
        if high > 0:
            st.error(
                "High-risk items detected. Immediate removal recommended."
            )
        elif mid > 0:
            st.warning(
                "Medium-risk items present. Sell soon or apply discounts."
            )
        else:
            st.success(
                "Batch quality looks good. Normal inventory handling is safe."
            )

        # Downloadable report
        report_name = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        st.download_button(
            "Download Batch Report",
            data=df.to_csv(index=False),
            file_name=report_name,
            mime="text/csv"
        )



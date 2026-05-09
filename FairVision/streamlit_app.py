import streamlit as st
import torch
from PIL import Image
from collections import Counter
import pandas as pd
import altair as alt
from MLModel import CNN, eval_transform

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="FairVision AI",
    layout="wide",
    page_icon="🧠"
)

# =========================
# STYLING
# =========================
st.markdown("""
<style>
    .title {
        text-align:center;
        font-size:42px;
        font-weight:bold;
        color:#4A90E2;
    }
    .card {
        background-color:#111827;
        padding:12px;
        border-radius:12px;
        text-align:center;
        margin-top:8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LABELS
# =========================
ages = [
    "0-2","3-9","10-19","20-29","30-39",
    "40-49","50-59","60-69","70+"
]

# =========================
# MODEL
# =========================
model = CNN(len(ages))
checkpoint = torch.load("best_fairface_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = eval_transform

# =========================
# HEADER
# =========================
st.markdown('<div class="title">🧠 FairVision Age Classifier</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📤 Upload Face Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================
# STORAGE
# =========================
predictions = []

# =========================
# GRID DISPLAY (9 per row fix)
# =========================
if uploaded_files:

    cols = st.columns(9)   # ⭐ 5 images per row

    for idx, file in enumerate(uploaded_files):

        img = Image.open(file).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            pred_class = int(torch.argmax(probs, dim=1))
            confidence = float(probs[0][pred_class])

            predicted_age = ages[pred_class]

        predictions.append(predicted_age)

        with cols[idx % 9]:
            st.image(img, use_container_width=True)

            st.markdown(
                f"""
                <div class="card">
                    <h3 style="color:#00C853;">🎯 {predicted_age}</h3>
                    <p style="color:#90CAF9;">Confidence: {confidence:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# TAB VIEW
# =========================
if uploaded_files and len(uploaded_files) > 0:

    tab1, tab2, tab3 = st.tabs([
        "📊 Age Analysis",
        "🌍 Map View",
        "🚻 Gender Analysis"
    ])

    # =========================
    # 📊 AGE TAB (FIXED)
    # =========================
    with tab1:

        st.markdown("### 📊 Age Distribution")

        if len(predictions) == 0:
            st.info("No predictions yet")
        else:
            count = Counter(predictions)

            # keep fixed order (IMPORTANT FIX)
            df = pd.DataFrame({
                "Age Group": ages,
                "Count": [count.get(a, 0) for a in ages]
            })

            chart = alt.Chart(df).mark_line(point=True).encode(
                x="Age Group",
                y="Count"
            ).properties(width=700)

            st.altair_chart(chart, use_container_width=True)

    # =========================
    # 🌍 MAP TAB (placeholder safe)
    # =========================
    with tab2:
        st.markdown("### 🌍 Map View")
        st.info("Add map_data here (not included in your snippet)")

    # =========================
    # 🚻 GENDER TAB (ICON COLOR STYLE)
    # =========================
    with tab3:

        st.markdown("### 🌍 Map View")
        st.info("Add gender_data here (not included in your snippet)")
import streamlit as st
import torch
from PIL import Image
from MLModel import CNN, eval_transform

ages = [
    "0-2","3-9","10-19","20-29","30-39",
    "40-49","50-59","60-69","70+"
]

model = CNN(len(ages))
checkpoint = torch.load("best_fairface_model.pt", map_location="cpu")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = eval_transform

st.title("🧠 FairVision Age Classification")

uploaded_files = st.file_uploader(
    "Upload Face Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        st.image(img, caption=file.name, use_container_width=True)

        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            predicted_age = ages[pred_class]

        st.write(f"🎯 **Age Group:** {predicted_age}")
        st.write(f"📊 **Confidence:** {confidence:.2f}")
        st.markdown("---")
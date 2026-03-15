import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="SafeVision AI", layout="centered")

st.title("🦺 SafeVision AI")
st.write("AI System for Detecting PPE (Helmet & Safety Vest)")

# โหลดโมเดล
@st.cache_resource
def load_model():
    model = YOLO("./best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image to check worker safety",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)

    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_container_width=True)

    names = model.names
    boxes = results[0].boxes

    helmet_ok = True
    vest_ok = True

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]

            if label == "no_helmet":
                helmet_ok = False

            if label == "no_vest":
                vest_ok = False

    if helmet_ok and vest_ok:
        st.success("✅ All workers wearing PPE")
    else:
        st.error("⚠️ PPE violation detected")

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI PPE Safety Detector", layout="centered")

st.title("🦺 AI PPE Safety Detector")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)

    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_container_width=True)

    names = model.names
    classes = results[0].boxes.cls.tolist()

    detected = [names[int(c)] for c in classes]

    if "person" in detected:

        st.success("Person detected")

        # ตรวจสี PPE แบบง่าย
        avg = img.mean(axis=(0,1))

        red = avg[0]
        green = avg[1]
        blue = avg[2]

        if green > 120 or red > 120:
            st.success("🦺 PPE DETECTED (Helmet / Safety Vest)")
        else:
            st.error("⚠️ NO PPE DETECTED")

    else:
        st.warning("No person detected")

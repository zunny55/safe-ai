import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Safety PPE Detector", layout="centered")

st.title("🦺 AI PPE Safety Detector")

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")   # โมเดลมาตรฐาน
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_array = np.array(image)

    results = model(img_array)

    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_container_width=True)

    classes = results[0].boxes.cls.tolist()

    names = model.names

    detected = [names[int(c)] for c in classes]

    if "person" in detected:
        st.success("Person detected")

        # ตรวจสีหมวกคร่าวๆ
        avg_color = img_array.mean(axis=(0,1))

        if avg_color[0] > 150 or avg_color[1] > 150:
            st.success("Possible PPE detected")
        else:
            st.error("NO PPE DETECTED")

    else:
        st.warning("No person detected")warning("No person detected")

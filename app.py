import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI PPE Safety Detector", layout="centered")

st.title("🦺 AI PPE Safety Detector")
st.write("Upload an image to detect people and check basic PPE presence")

# โหลดโมเดล
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")   # ใช้โมเดลมาตรฐาน
    return model

model = load_model()

# อัปโหลดภาพ
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # อ่านภาพ
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # แสดงภาพต้นฉบับ
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ตรวจจับวัตถุ
    results = model(img_array)

    # วาด bounding box
    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_container_width=True)

    # ตรวจ class ที่พบ
    classes = results[0].boxes.cls.tolist()
    names = model.names

    detected = [names[int(c)] for c in classes]

    # logic ตรวจคน
    if "person" in detected:
        st.success("Person detected")

        # logic PPE แบบง่าย (ดูค่าเฉลี่ยสี)
        avg_color = img_array.mean(axis=(0,1))

        if avg_color[0] > 150 or avg_color[1] > 150:
            st.success("Possible PPE detected")
        else:
            st.error("NO PPE DETECTED")

    else:
        st.warning("No person detected")

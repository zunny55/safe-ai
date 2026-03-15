import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="SafeVision AI", layout="centered")

st.title("🦺 SafeVision AI")
st.write("AI ตรวจสอบ PPE (Helmet / Safety Vest)")

# แสดงไฟล์ในโฟลเดอร์เพื่อ debug
st.write("Files in project:", os.listdir())

# โหลดโมเดล
@st.cache_resource
def load_model():
    model = YOLO("./best.pt")
    return model

model = load_model()

# แสดง class ของโมเดล
st.write("Model classes:", model.names)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detection Result", use_container_width=True)

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

    if not helmet_missing and not vest_missing:
        st.success("✅ All workers wearing PPE")

    st.image(frame, channels="BGR")

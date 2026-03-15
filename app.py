import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="PPE Detector")

st.title("🦺 PPE Safety Detector")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file)
    img = np.array(image)

    results = model.predict(img)

    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_container_width=True)

    names = model.names
    classes = results[0].boxes.cls.tolist()

    detected = [names[int(c)] for c in classes]

    if "person" in detected:

        st.success("Person detected")

        # ตรวจสี PPE
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # สีเหลือง (helmet / vest)
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([35,255,255])

        # สีส้ม (vest)
        lower_orange = np.array([5,100,100])
        upper_orange = np.array([15,255,255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

        yellow_pixels = np.sum(mask_yellow > 0)
        orange_pixels = np.sum(mask_orange > 0)

        if yellow_pixels > 5000:
            st.success("🪖 Helmet detected")

        if orange_pixels > 5000:
            st.success("🦺 Safety vest detected")

        if yellow_pixels < 5000 and orange_pixels < 5000:
            st.error("⚠️ No PPE detected")

    else:
        st.warning("No person detected")

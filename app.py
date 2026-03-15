import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="PPE Detector")

st.title("🦺 PPE Detection Demo")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # แปลงเป็น HSV
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

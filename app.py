import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🦺 PPE Detector")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    h, w, _ = img.shape

    # ตัดเฉพาะช่วงลำตัว (ประมาณ 30% ถึง 70% ของภาพ)
    torso = img[int(h*0.3):int(h*0.7), int(w*0.25):int(w*0.75)]

    hsv = cv2.cvtColor(torso, cv2.COLOR_RGB2HSV)

    # สีเหลือง
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([35,255,255])

    # สีส้ม
    lower_orange = np.array([5,100,100])
    upper_orange = np.array([15,255,255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    yellow_pixels = np.sum(mask_yellow > 0)
    orange_pixels = np.sum(mask_orange > 0)

    if yellow_pixels > 2000 or orange_pixels > 2000:
        st.success("🦺 Safety vest detected")
    else:
        st.error("⚠️ No safety vest detected")
